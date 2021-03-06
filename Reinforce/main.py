import argparse

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from TorchPolicyGrads.Reinforce.Reinforce import Reinforce
from TorchPolicyGrads.Reinforce.ReinforceBaseline import ReinforceBaseline

parser = argparse.ArgumentParser(description='Reinforce')
parser.add_argument('--lr', type=float, default=.0025,
                    help='learning rate (default: 0.0025)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-episodes', type=int, default=2500,
                    help='number of episodes (default: 250)')
parser.add_argument('--max-episode-length', type=int, default=250,
                    help='maximum length of an episode (default: 250)')
parser.add_argument('--env-name', default='CartPole-v0',
                    help='environment to train on (default: CartPole-v0)')
parser.add_argument('--hidden-units', default=[12], type=list,
                    help='hidden units in policy (default: [12])')
parser.add_argument('--target', default=195, type=int,
                    help='Solved at target (default: 195)')
parser.add_argument('--baseline', default=True, type=bool,
                    help='use reinforce with baseline (default: false)')
parser.add_argument('--window-size', default=100, type=int,
                    help='Window size to calculate variance (default: 100)')


def plot_rewards(reinforce, episode_rewards, mean_rewards, env_name, num_episodes):
    episodes = np.arange(num_episodes)
    fig = plt.figure()
    plt.title(f"{type(reinforce).__name__}: {env_name} Rewards over {num_episodes} episodes")
    plt.plot(episodes, episode_rewards, label="episode reward")
    plt.plot(episodes, mean_rewards, label="mean reward")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    fig.savefig(f'{type(reinforce).__name__}-rewards.png')


def plot_variance(episode_rewards_reinforce, episode_rewards_baseline, env_name, window_size, num_episodes):
    assert num_episodes % window_size == 0, f"window_size: {window_size} does not divide {num_episodes}"
    fig = plt.figure()
    plt.title(f"{env_name} Variance (sliding window of {window_size} episodes) over {num_episodes} episodes")
    reinforce_var = np.var(rolling_window(np.array(episode_rewards_reinforce), window_size),-1)
    reinforce_baseline_var = np.var(rolling_window(np.array(episode_rewards_baseline), window_size),-1)
    windows = np.arange(len(reinforce_var))
    plt.plot(windows,reinforce_var,label="Reinforce")
    plt.plot(windows, reinforce_baseline_var, label="Reinforce-Baseline")
    plt.ylabel("Variance")
    plt.xlabel("Episode")
    plt.legend()
    fig.savefig('variance.png')


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = gym.make(args.env_name)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    reinforce = Reinforce(state_dim=obs_shape, n_actions=n_actions, hidden_units=args.hidden_units, lr=args.lr)
    reinforceBaseline = ReinforceBaseline(state_dim=obs_shape, n_actions=n_actions, hidden_units=args.hidden_units,
                                          lr=args.lr)

    agent_episode_rewards = []

    for agent in [reinforce, reinforceBaseline]:
        episode_rewards, mean_rewards = agent(env, args.num_episodes, args.gamma, args.max_episode_length, args.target)
        agent_episode_rewards.append(episode_rewards)
        plot_rewards(agent, episode_rewards, mean_rewards, args.env_name, args.num_episodes)
        agent.demo(env)

    episode_rewards_reinforce, episode_rewards_baseline = agent_episode_rewards
    plot_variance(episode_rewards_reinforce, episode_rewards_baseline, args.env_name, args.window_size,
                  args.num_episodes)
