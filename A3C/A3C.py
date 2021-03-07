import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from SharedOptim import SharedAdam
from utils.misc import compute_discounted_rewards

device = 'cpu'
huber_loss = nn.SmoothL1Loss(reduction='sum')


def compute_loss(action_probs, rewards, values):
    log_action_probs = torch.log(action_probs)
    advs = rewards - values
    return -torch.sum(log_action_probs * advs)


class ACNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_units):
        super(ACNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], hidden_units[0])
        self.fc2 = nn.Linear(state_dim[0], hidden_units[0])
        self.fc3 = nn.Linear(hidden_units[0], n_actions)
        self.fc4 = nn.Linear(hidden_units[0], 1)

    def forward(self, inputs):
        action_hidden = F.relu(self.fc1(inputs))
        action_logits = self.fc3(action_hidden)
        state_hidden = F.relu(self.fc2(inputs))
        state_value = self.fc4(state_hidden)
        return action_logits, state_value

    def update_weights_from_master(self, master_net):
        self.load_state_dict(master_net.state_dict())


class Worker:
    def __init__(self, env, rank, hidden_units):
        self.env = env
        self.rank = rank
        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        self.network = ACNetwork(obs_shape, n_actions, hidden_units)

    def run_episode(self, max_episode_length):
        rewards, action_probs, state_values = [], [], []
        state = self.env.reset()
        for step in range(max_episode_length):
            state = torch.FloatTensor(state)
            action_logits, state_value = self.network(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).cpu().detach().numpy()
            action_prob = distr.probs[action]
            state, reward, done, _ = self.env.step(action)

            rewards.append(reward)
            action_probs.append(action_prob)
            state_values.append(state_value)

            if done:
                break
        return torch.Tensor(rewards), torch.stack(action_probs), torch.stack(state_values)

    def work(self, master_network, master_optim, global_episodes, global_avg_reward, global_avg_reward_arr,
             max_episodes, max_episode_length,
             gamma, target):
        print(f"Starting worker {self.rank}")
        while True:
            with global_episodes.get_lock(), global_avg_reward.get_lock():
                if global_episodes.value >= max_episodes or global_avg_reward.value >= target:
                    break

            self.network.update_weights_from_master(master_network)
            master_optim.zero_grad()
            rewards, action_probs, state_values = self.run_episode(max_episode_length)
            state_values = torch.squeeze(state_values)
            discounted_rewards = compute_discounted_rewards(rewards, gamma, device, normalize_rewards=False)
            loss = torch.mean(
                compute_loss(action_probs, discounted_rewards, state_values) + huber_loss(discounted_rewards,
                                                                                          state_values))
            loss.backward()

            master_optim.step()

            episode_reward = np.sum(rewards.detach().numpy()).item()

            with global_avg_reward.get_lock(), global_episodes.get_lock():
                if global_episodes.value >= max_episodes:
                    break
                global_avg_reward.value = global_avg_reward.value * .99 + episode_reward * .01
                with global_avg_reward_arr.get_lock():
                    global_avg_reward_arr[global_episodes.value] = global_avg_reward.value
                    global_episodes.value += 1
        print(f"Exiting worker {self.rank}")


class A3C:
    def __init__(self, env_name, num_workers, hidden_units, lr):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.num_workers = num_workers
        self.hidden_units = hidden_units
        self.lr = lr

        obs_shape = self.env.observation_space.shape
        n_actions = self.env.action_space.n

        self.global_network = ACNetwork(obs_shape, n_actions, self.hidden_units)
        self.global_optim = SharedAdam(self.global_network.parameters(), lr=lr)

    def train(self, seed, max_episodes, gamma, max_episode_length, target):
        torch.manual_seed(seed)

        global_episodes = mp.Value('i', 0)
        global_avg_reward = mp.Value('f', 0)
        global_avg_reward_arr = mp.Array('f', range(max_episodes))
        self.global_network.share_memory()

        workers = []
        for i in range(self.num_workers):
            workers.append(
                Worker(gym.make(self.env_name), i, self.hidden_units)
            )
        processes = []
        for worker in workers:
            p = mp.Process(target=worker.work,
                           args=(
                               self.global_network, self.global_optim, global_episodes, global_avg_reward,
                               global_avg_reward_arr, max_episodes, max_episode_length, gamma, target))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        return global_avg_reward_arr

    def test(self, seed):
        torch.manual_seed(seed)
        state = self.env.reset()
        while True:
            state = torch.FloatTensor(state)
            action_logits, _ = self.global_network(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).detach().numpy()
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            if done:
                return
