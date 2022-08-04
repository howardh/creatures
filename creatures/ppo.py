from typing import Optional, Generator, Dict
from torchtyping import TensorType
import time

import gym
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import wandb

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss


class Model(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.ReLU(),
        )
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)
    def forward(self, x):
        x = self.conv(x)
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi, # Unnormalized action probabilities
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
        observation_space : gym.Space,
        action_space : gym.Space,
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
        num_minibatches : int) -> Generator[Dict[str,TensorType],None,None]:
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
        action_dist = torch.distributions.Categorical(logits=net_output['action'][:n-1])
        log_action_probs_old = action_dist.log_prob(action)

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
        action_dist = torch.distributions.Categorical(logits=net_output['action'])
        mb_log_action_probs = action_dist.log_prob(mb_action)
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
                epsilon=0.1
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


def train_ppo_atari(
        model: torch.nn.Module,
        env: gym.vector.VectorEnv,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # XXX: Private class. This might break in the future.
        *,
        max_steps: int = 1000,
        rollout_length: int = 128,
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = 1.0,
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

    obs = env.reset()
    history.append_obs(obs)
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    env_steps = 0
    for step in range(max_steps):
        # Gather data
        for i in range(rollout_length):
            env_steps += num_envs

            # Select action
            with torch.no_grad():
                action_probs = model(torch.tensor(obs, dtype=torch.float, device=device))['action'].softmax(1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, done, info = env.step(action)

            history.append_action(action)
            episode_reward += reward
            episode_steps += 1

            reward *= reward_scale
            if reward_clip is not None:
                reward = np.clip(reward, -reward_clip, reward_clip)

            history.append_obs(obs, reward, done)

            if done.any():
                if 'lives' in info:
                    done = info['lives'] == 0
            if done.any():
                print(f'{step * num_envs * rollout_length:,}\t reward: {episode_reward[done].mean():.2f}\t len: {episode_steps[done].mean()}')
                wandb.log({
                        'reward': episode_reward[done].mean().item(),
                        'episode_length': episode_steps[done].mean().item(),
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

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
            steps_per_sec = step / elapsed_time
            remaining_time = int((max_steps - step) / steps_per_sec)
            remaining_hours = remaining_time // 3600
            remaining_minutes = (remaining_time % 3600) // 60
            remaining_seconds = (remaining_time % 3600) % 60
            print(f"Step {step:,}/{max_steps:,} \t {int(steps_per_sec * num_envs * rollout_length):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")

        yield


if __name__ == '__main__':
    import argparse
    from gym.wrappers import AtariPreprocessing, FrameStack

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='Environment to train on')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of environments to train on')
    parser.add_argument('--max-steps', type=int, default=10_000_000//8//128, help='Number of training steps to run. One step is one weight update.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--rollout-length', type=int, default=128, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=None, help='Clip the reward magnitude to this value.')
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--norm-adv', type=bool, default=True, help='Normalize the advantages.')
    parser.add_argument('--clip-vf-loss', type=bool, default=True, help='Clip the value function loss.')
    parser.add_argument('--vf-loss-coeff', type=float, default=0.5, help='Coefficient for the value function loss.')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0.01, help='Coefficient for the entropy loss.')
    parser.add_argument('--target-kl', type=float, default=0.01, help='Target KL divergence.')
    parser.add_argument('--minibatch-size', type=int, default=256, help='Minibatch size.')
    parser.add_argument('--num-minibatches', type=int, default=4, help='Number of minibatches.')
    parser.add_argument('--max-grad-norm', type=float, default=None, help='Maximum gradient norm.')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')

    args = parser.parse_args()

    wandb.init(project='ppo-atari')
    wandb.config.update(args)

    def make_env(name):
        env = gym.make(name)
        env = AtariPreprocessing(env)
        env = FrameStack(env, 4)
        return env
    env = gym.vector.AsyncVectorEnv([lambda: make_env(args.env) for _ in range(args.num_envs)])

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = Model(num_actions=env.single_action_space.n)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = None

    trainer = train_ppo_atari(
            model = model,
            env = env,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            max_steps = args.max_steps,
            rollout_length = args.rollout_length,
            reward_clip = args.reward_clip,
            reward_scale = args.reward_scale,
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
