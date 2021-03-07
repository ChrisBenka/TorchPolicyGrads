import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(agent, episode_rewards, mean_rewards, env_name, num_episodes):
    episodes = np.arange(num_episodes)
    fig = plt.figure()
    plt.title(f"{type(agent).__name__}: {env_name} Rewards over {num_episodes} episodes")
    plt.plot(episodes, episode_rewards, label="episode reward")
    plt.plot(episodes, mean_rewards, label="mean reward")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    fig.savefig(f'{type(agent).__name__}-rewards.png')

def plot_mean_rewards(agent, mean_rewards, env_name, num_episodes):
    episodes = np.arange(num_episodes)
    fig = plt.figure()
    plt.title(f"{type(agent).__name__}: {env_name} Rewards over {num_episodes} episodes")
    plt.plot(episodes, mean_rewards, label="mean reward")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.legend()
    fig.savefig(f'{type(agent).__name__}-rewards.png')


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
