import argparse

from A3C import A3C
from Utils.plot import plot_mean_rewards

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=13,
                    help='random seed (default: 13)')
parser.add_argument('--num-episodes', type=int, default=10000,
                    help='number of episodes (default: 10000)')
parser.add_argument('--max-episode-length', type=int, default=250,
                    help='maximum length of an episode (default: 250)')
parser.add_argument('--env-name', default='CartPole-v0',
                    help='environment to train.py on (default: CartPole-v0)')
parser.add_argument('--target', default=195, type=int,
                    help='Solved at target (default: 195)')
parser.add_argument('--num-workers', default=20, type=int,
                    help='number of threads to be used during training (defaults: 20)')
parser.add_argument('--update-interval', type=int, default=10,
                    help='interval to synchronize local to global (default:10)')

if __name__ == '__main__':
    args = parser.parse_args()
    agent = A3C(env_name=args.env_name, n_workers=args.num_workers, lr=args.lr)
    episode_rewards, mean_rewards = agent.train(args.seed, args.num_episodes, args.gamma,
                                                args.max_episode_length, args.target, args.update_interval)
    plot_mean_rewards(agent, mean_rewards, args.env_name, len(episode_rewards))
    agent.test()
