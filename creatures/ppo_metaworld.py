import itertools
import random
from typing import Optional, Generator, Dict, Any
import time
import warnings

import gymnasium
from gymnasium.vector import VectorEnv, AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, TimeLimit # pyright: ignore[reportPrivateImportUsage]
from gymnasium.wrappers import StepAPICompatibility # pyright: reportPrivateImportUsage=false
from gymnasium.utils.step_api_compatibility import step_api_compatibility
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import wandb
import metaworld

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss


def _convert_space_gym_to_gymnasium(space):
    # No need to do anything if it's already a gymnasium space
    if isinstance(space, gymnasium.spaces.Space):
        return space

    if type(space).__name__ == 'Box':
        return gymnasium.spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
    elif type(space).__name__ == 'Discrete':
        return gymnasium.spaces.Discrete(n=space.n)
    elif type(space).__name__ == 'Tuple':
        return gymnasium.spaces.Tuple(tuple(_convert_space_gym_to_gymnasium(s) for s in space.spaces))
    elif type(space).__name__ == 'Dict':
        return gymnasium.spaces.Dict(dict((k, _convert_space_gym_to_gymnasium(s)) for k, s in space.spaces.items()))
    else:
        raise NotImplementedError(f"Unsupported space type {type(space).__name__}")


class GymCompatibilityWrapper(StepAPICompatibility):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = _convert_space_gym_to_gymnasium(env.action_space)
        self.observation_space = _convert_space_gym_to_gymnasium(env.observation_space)

    def reset(self, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, 'seed') and callable(self.env.seed):
                self.env.seed(seed)
            else:
                warnings.warn("Seed was provided but the environment does not have a seed() method.")

        if options is not None:
            warnings.warn("Options were provided but we don't know how to handle them because there was no standard way to do so in OpenAI Gym.")

        return self.env.reset(), {}

    def step(self, action):
        step_returns = self.env.step(action)
        _, _, done, _ = step_returns
        if done:
            breakpoint()
        return step_api_compatibility(
            step_returns, self.output_truncation_bool, self.is_vector_env
        )


class Model(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.v = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128,out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128,out_features=1),
        )
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64,out_features=64),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=64,out_features=action_size),
        )
        self.log_std = torch.nn.Parameter(torch.zeros(action_size))
    def forward(self, x):
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi,
                'action_mean': pi,
                'action_logstd': self.log_std[None].expand_as(pi),
        }


def enum_minibatches(batch_size, minibatch_size, num_minibatches, replace=False):
    indices = np.arange(batch_size)
    if replace:
        for _ in range(0,batch_size,minibatch_size):
            np.random.shuffle(indices)
            yield indices[:minibatch_size]
    else:
        indices = np.arange(batch_size)
        n = batch_size//minibatch_size
        for i in range(num_minibatches):
            j = i % n
            if j == 0:
                np.random.shuffle(indices)
            yield indices[j*minibatch_size:(j+1)*minibatch_size]


def compute_ppo_losses(
        observation_space : gymnasium.Space,
        action_space : gymnasium.Space,
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        discount : float,
        gae_lambda : float,
        norm_adv : bool,
        clip_vf_loss : Optional[float],
        entropy_loss_coeff : float,
        vf_loss_coeff : float,
        target_kl : Optional[float],
        minibatch_size : int,
        num_minibatches : int) -> Generator[Dict[str,Any],None,None]:
    """
    Compute the losses for PPO.
    """
    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal

    n = len(history.obs_history)
    num_training_envs = len(history.obs_history[0])

    with torch.no_grad():
        net_output = default_collate([model(torch.tensor(o, dtype=torch.float, device=device)) for o in obs])
        state_values_old = net_output['value'].squeeze(2)
        action_mean = net_output['action_mean'][:n-1]
        action_logstd = net_output['action_logstd'][:n-1]
        action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
        log_action_probs_old = action_dist.log_prob(action).sum(-1)

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values_old[:n-1,:],
                next_state_values = state_values_old[1:,:],
                rewards = reward[1:,:],
                terminals = terminal[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )
        returns = advantages + state_values_old[:n-1,:]

    # Flatten everything
    flat_obs = obs[:n-1].reshape(-1,*observation_space.shape)
    flat_action = action[:n-1].reshape(-1, *action_space.shape)
    flat_terminals = terminal[:n-1].reshape(-1)
    flat_returns = returns[:n-1].reshape(-1)
    flat_advantages = advantages[:n-1].reshape(-1)
    flat_log_action_probs_old = log_action_probs_old[:n-1].reshape(-1)
    flat_state_values_old = state_values_old[:n-1].reshape(-1)

    minibatches = enum_minibatches(
            batch_size=(n-1) * num_training_envs,
            minibatch_size=minibatch_size,
            num_minibatches=num_minibatches,
            replace=False)

    for _,mb_inds in enumerate(minibatches):
        mb_obs = torch.tensor(flat_obs[mb_inds], dtype=torch.float, device=device)
        mb_action = flat_action[mb_inds]
        mb_returns = flat_returns[mb_inds]
        mb_advantages = flat_advantages[mb_inds]
        mb_terminals = flat_terminals[mb_inds]
        mb_log_action_probs_old = flat_log_action_probs_old[mb_inds]
        mb_state_values_old = flat_state_values_old[mb_inds]

        net_output = model(mb_obs)
        assert 'value' in net_output
        assert 'action' in net_output
        mb_state_values = net_output['value'].squeeze()
        mb_action_mean = net_output['action_mean']
        mb_action_logstd = net_output['action_logstd']
        action_dist = torch.distributions.Normal(mb_action_mean, mb_action_logstd.exp())
        mb_log_action_probs = action_dist.log_prob(mb_action).sum(-1)
        mb_entropy = action_dist.entropy()

        with torch.no_grad():
            logratio = mb_log_action_probs - mb_log_action_probs_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss = clipped_advantage_policy_gradient_loss(
                log_action_probs = mb_log_action_probs,
                old_log_action_probs = mb_log_action_probs_old,
                advantages = mb_advantages,
                terminals = mb_terminals,
                epsilon=0.2
        ).mean()

        # Value loss
        if clip_vf_loss is not None:
            v_loss_unclipped = (mb_state_values - mb_returns) ** 2
            v_clipped = mb_state_values_old + torch.clamp(
                mb_state_values - mb_state_values_old,
                -clip_vf_loss,
                clip_vf_loss,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((mb_state_values - mb_returns) ** 2).mean()

        entropy_loss = mb_entropy.mean()
        loss = pg_loss - entropy_loss_coeff * entropy_loss + v_loss * vf_loss_coeff

        yield {
                'loss': loss,
                'loss_pi': pg_loss,
                'loss_vf': v_loss,
                'loss_entropy': -entropy_loss,
                'approx_kl': approx_kl,
                'state_value': mb_state_values,
                'entropy': mb_entropy,
        }

        if target_kl is not None:
            if approx_kl > target_kl:
                break


def train_ppo_metaworld(
        model: torch.nn.Module,
        env: VectorEnv,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # XXX: Private class. This might break in the future.
        *,
        max_steps: int = 1000,
        rollout_length: int = 128,
        max_grad_norm: float = 0.5,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_vf_loss: Optional[float] = None,
        entropy_loss_coeff: float = 0.01,
        vf_loss_coeff: float = 0.5,
        num_minibatches: int = 4,
        minibatch_size: int = 32,
        target_kl: Optional[float] = None,
        norm_adv: bool = True,
        ):
    """
    Train a model with PPO on an Atari game.

    Args:
        model: ...
        env: `gym.vector.VectorEnv`
    """
    num_envs = env.num_envs
    device = next(model.parameters()).device
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)
    start_time = time.time()

    obs, _ = env.reset()
    history.append_obs(obs)
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    env_steps = 0
    for step in itertools.count():
        # Gather data
        for i in range(rollout_length):
            env_steps += num_envs

            # Select action
            with torch.no_grad():
                model_output = model(torch.tensor(obs, dtype=torch.float, device=device))
                action_mean = model_output['action_mean']
                action_logstd = model_output['action_logstd']
                action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
                action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated | truncated

            history.append_action(action)
            episode_reward += reward
            episode_steps += 1

            history.append_obs(obs, reward, done)

            if done.any():
                ep_rew = np.array([x['episode']['r'] for x in info['final_info'] if x is not None])
                ep_len = np.array([x['episode']['l'] for x in info['final_info'] if x is not None])
                print(f'{step * num_envs * rollout_length:,}\t reward: {ep_rew.mean():.2f}\t len: {ep_len.mean()} \t ({done.sum()} done)')
                if wandb.run is not None:
                    wandb.log({
                            'reward': ep_rew.mean().item(),
                            'episode_length': ep_len.mean().item(),
                            'step': env_steps,
                    }, step = env_steps)
                episode_reward[done] = 0
                episode_steps[done] = 0

        # Train
        losses = compute_ppo_losses(
                observation_space = observation_space,
                action_space = action_space,
                history = history,
                model = model,
                discount = discount,
                gae_lambda = gae_lambda,
                norm_adv = norm_adv,
                clip_vf_loss = clip_vf_loss,
                vf_loss_coeff = vf_loss_coeff,
                entropy_loss_coeff = entropy_loss_coeff,
                target_kl = target_kl,
                minibatch_size = minibatch_size,
                num_minibatches = num_minibatches,
        )
        for i,x in enumerate(losses):
            optimizer.zero_grad()
            x['loss'].backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # pyright: ignore[reportPrivateImportUsage]
            optimizer.step()

            if wandb.run is not None:
                wandb.log({
                    f'loss/pi/{i}': x['loss_pi'].item(),
                    f'loss/v/{i}': x['loss_vf'].item(),
                    f'loss/entropy/{i}': x['loss_entropy'].item(),
                    f'loss/total/{i}': x['loss'].item(),
                    f'approx_kl/{i}': x['approx_kl'].item(),
                    f'state_value/{i}': x['state_value'].mean().item(),
                    f'entropy/{i}': x['entropy'].mean().item(),
                    #last_approx_kl=approx_kl.item(),
                    #'learning_rate': lr_scheduler.get_lr()[0],
                    'step': env_steps,
                }, step = env_steps)

        # Clear data
        history.clear()

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Timing
        if step > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = env_steps / elapsed_time
            if max_steps > 0:
                remaining_time = int((max_steps - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Step {env_steps:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            else:
                elapsed_time = int(elapsed_time)
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                elapsed_seconds = (elapsed_time % 3600) % 60
                print(f"Step {env_steps:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")

        if max_steps > 0 and env_steps >= max_steps:
            break

        yield


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train MetaWorld with PPO. Default parameters taken from https://arxiv.org/pdf/1910.10897.pdf Appendix D.2 (Single task PPO).')

    parser.add_argument('--env', type=str, default='pick-place-v2', help='Environment to train on')
    parser.add_argument('--num-envs', type=int, default=10, help='Number of environments to train on') # XXX: Not sure about this one. Batch size is 5000 and rollout length is 500, so I'm guessing it means 10 rollouts?
    parser.add_argument('--max-steps', type=int, default=4_000, help='Number of training steps to run. One step is one weight update. If 0, train forever.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer', choices=['Adam', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=5.0e-4, help='Learning rate.')
    parser.add_argument('--rollout-length', type=int, default=500, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=10, help='Clip the reward magnitude to this value.') # CleanRL uses 10
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--norm-adv', type=bool, default=True, help='Normalize the advantages.')
    parser.add_argument('--clip-vf-loss', type=float, default=0.2, help='Clip the value function loss.')
    parser.add_argument('--vf-loss-coeff', type=float, default=0.5, help='Coefficient for the value function loss.')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0.0, help='Coefficient for the entropy loss.')
    parser.add_argument('--target-kl', type=float, default=None, help='Target KL divergence.')
    parser.add_argument('--minibatch-size', type=int, default=64, help='Minibatch size.')
    parser.add_argument('--num-minibatches', type=int, default=10, help='Number of minibatches.')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm.')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases.')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='ppo-metaworld')
        wandb.config.update(args, allow_val_change=True)

    def make_env(name):
        ml1 = metaworld.ML1(name)
        env = ml1.train_classes[name]()
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env = GymCompatibilityWrapper(env)
        env = TimeLimit(env, max_episode_steps=env.max_path_length)

        env = RecordEpisodeStatistics(env)

        env = ClipAction(env)

        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        env = NormalizeReward(env, gamma=args.discount)
        env = TransformReward(env, lambda reward: np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip))

        return env
    env = SyncVectorEnv([lambda: make_env(args.env) for _ in range(args.num_envs)])

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    assert env.single_observation_space.shape is not None
    assert env.single_action_space.shape is not None
    model = Model(
            obs_size=env.single_observation_space.shape[0],
            action_size=env.single_action_space.shape[0]
    )
    model.to(device)
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    lr_scheduler = None

    trainer = train_ppo_metaworld(
            model = model,
            env = env,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            max_steps = args.max_steps,
            rollout_length = args.rollout_length,
            discount = args.discount,
            gae_lambda = args.gae_lambda,
            norm_adv = args.norm_adv,
            clip_vf_loss = args.clip_vf_loss,
            vf_loss_coeff = args.vf_loss_coeff,
            entropy_loss_coeff = args.entropy_loss_coeff,
            target_kl = args.target_kl,
            minibatch_size = args.minibatch_size,
            num_minibatches = args.num_minibatches,
            max_grad_norm = args.max_grad_norm,
    )
    for _ in trainer:
        pass
