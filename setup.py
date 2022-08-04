from setuptools import setup, find_packages

setup(name='creatures',
    version='0.0.1',
    install_requires=['frankenstein'],
    extras_require={
        'dev': ['pytest','pytest-cov','pytest-timeout','pdoc','numpy','flake8','autopep8'],
    },
    packages=find_packages()
)
