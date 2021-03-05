import argparse

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from TorchPolicyGrads.Reinforce.Reinforce import Reinforce

parser = argparse.ArgumentParser(description='A3C')
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
parser.add_argument('--hidden-units', default=[12, 128], type=list,
                    help='hidden units in policy (default: [256,128])')
parser.add_argument('--target', default=195, type=int,
                    help='Solved at target (default: 195)')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    env = gym.make(args.env_name)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    reinforce = Reinforce(state_dim=obs_shape, n_actions=n_actions, hidden_units=args.hidden_units, lr=args.lr)
    episode_rewards, mean_rewards = reinforce(env, args.num_episodes, args.gamma, args.max_episode_length, args.target)

    reinforce.demo(env)

    episodes = np.arange(len(episode_rewards))
    plt.title(f"Reinforce: {args.env_name} over {args.num_episodes} episodes")
    plt.plot(episodes, episode_rewards, label="episode reward")
    plt.plot(episodes, mean_rewards, label="mean reward")
    plt.xlabel("Reward")
    plt.ylabel("episode")
    plt.legend()
    plt.savefig('plt.png')
