import argparse

from A3C import A3C
from utils.plot import plot_mean_rewards

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
                    help='environment to train.py on (default: CartPole-v0)')
parser.add_argument('--hidden-units', default=[12], type=list,
                    help='hidden units in policy (default: [12])')
parser.add_argument('--target', default=195, type=int,
                    help='Solved at target (default: 195)')
parser.add_argument('--num-workers', default=4, type=int,
                    help='number of threads to be used during training')
parser.add_argument('--window-size', default=100, type=int,
                    help='Window size to calculate variance (default: 100)')

if __name__ == '__main__':
    args = parser.parse_args()

    agent = A3C(env_name=args.env_name, num_workers=args.num_workers, hidden_units=args.hidden_units, lr=args.lr)
    mean_episode_rewards = agent.train(args.seed, args.num_episodes, args.gamma,
                                       args.max_episode_length, args.target)
    plot_mean_rewards(agent, mean_episode_rewards, args.env_name, args.num_episodes)
    agent.test(args.seed)
6
