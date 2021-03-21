import datetime
import time

import gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Categorical

from EnvUtils.AtariGymWapper import AtariGymWrapper
from SharedOptim import SharedAdam

PATH = "a3c_dict_model.pt"


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super(ActorCriticNet, self).__init__()
        self.conv1 = nn.Conv2d(4, out_channels=16, kernel_size=8, stride=4, )
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        # (state_dim - 32)/2 + 1
        self.fc1 = nn.Linear(2048, 256)
        self.fc_v = nn.Linear(256, 1)
        self.fc_actions = nn.Linear(256, n_actions)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 3, 1, 2)
        hidden = F.relu(self.conv1(inputs))
        hidden = F.relu(self.conv2(hidden))
        hidden_flat = hidden.view(-1, 32 * 8 * 8)
        hidden = F.relu(self.fc1(hidden_flat))
        return F.softmax(self.fc_actions(hidden),dim=-1), self.fc_v(hidden)

    def update_from_model(self, model: nn.Module):
        self.load_state_dict(model.state_dict())

    def get_action(self, obs):
        self.eval()
        probs, _ = self.forward(obs)
        distr = Categorical(probs=probs)
        action = distr.sample([1]).numpy()[0]
        return action

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, td_targets: torch.Tensor):
        self.train()
        probs, state_values = self.forward(states)
        advs = td_targets - state_values
        distr = Categorical(probs=probs)
        critic_loss = advs ** 2
        actor_loss = -(distr.log_prob(actions) * advs.detach().squeeze())
        return (critic_loss + actor_loss).mean()


class Worker:
    def __init__(self, rank: int, seed: int, env: gym.Env, gamma: float, update_interval: int):
        self.rank = rank
        self.env = env
        self.env.seed(rank + seed)
        state_dim = 84
        n_actions = self.env.action_space.n
        self.gamma = gamma
        self.update_interval = update_interval
        self.t_actor_critic = ActorCriticNet(state_dim=state_dim, n_actions=n_actions)

    def work(self, g_actor_critic: nn.Module, g_optim: SharedAdam, t_start, t_end, reward_queue: mp.Queue,
             mean_reward: mp.Value):
        print(f"starting {self.rank}")
        t_step_count = 0
        while True:
            self.t_actor_critic.update_from_model(g_actor_critic)
            episode_reward = 0
            state = self.env.reset()
            states, actions, rewards = [], [], []
            while True:
                action = self.t_actor_critic.get_action(torch.FloatTensor(state[None]))
                next_s, reward, done, _ = self.env.step(action)

                if self.rank == 0:
                    self.env.render()

                episode_reward += reward
                t_step_count += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if done or t_step_count % self.update_interval == 0:
                    if done:
                        state = self.env.reset()
                        with mean_reward.get_lock():
                            mean_reward.value = mean_reward.value * .99 + episode_reward * .01
                            print(
                                f"time_elapsed:{str(datetime.timedelta(seconds=time.time() - t_start))} worker:{self.rank} running_reward:{mean_reward.value}")
                        R = 0
                    else:
                        R = self.t_actor_critic(torch.FloatTensor(state[None]))[1].detach().numpy()[0]
                    td_targets = []
                    for reward in rewards[::-1]:
                        R = reward + self.gamma * R
                        td_targets.append(R)
                    td_targets = td_targets[::-1]
                    loss = self.t_actor_critic.compute_loss(torch.FloatTensor(states), torch.Tensor(actions),
                                                            torch.Tensor(td_targets))
                    g_optim.zero_grad()
                    loss.backward()
                    for t_param, g_param in zip(self.t_actor_critic.parameters(), g_actor_critic.parameters()):
                        g_param._grad = t_param.grad
                    g_optim.step()
                    states, actions, rewards = [], [], []
                else:
                    state = next_s

                if done:
                    break

            if time.time() > t_end:
                break
            else:
                reward_queue.put(episode_reward)

        reward_queue.put(None)
        print(f" exiting {self.rank}")


class A3C:
    def __init__(self, env_name, n_workers, lr):
        self.env_name = env_name
        self.n_workers = n_workers
        self.lr = lr
        self.env = AtariGymWrapper(gym.make(env_name))
        self.state_dim = 84
        self.n_actions = self.env.action_space.n
        self.global_network = ActorCriticNet(state_dim=self.state_dim, n_actions=self.n_actions)
        self.global_network.share_memory()
        self.global_optim = SharedAdam(params=self.global_network.parameters(), lr=lr)
        self.global_optim.share_memory()

    def train(self, seed, train_mins, gamma, update_interval):
        reward_queue, mean_reward = mp.Queue(), mp.Value('d', 0)
        workers, processes = [], []
        torch.manual_seed(seed)
        self.env.seed(seed)
        t_start = time.time()
        t_end = t_start + 60 * train_mins

        for rank in range(self.n_workers):
            worker = Worker(rank, seed, AtariGymWrapper(gym.make(self.env_name)), gamma, update_interval)
            p = mp.Process(target=worker.work,
                           args=(self.global_network, self.global_optim, t_start, t_end, reward_queue,
                                 mean_reward))
            p.start()
            processes.append(p)
            workers.append(worker)

        episode_rewards, mean_rewards = [], []
        running_reward = 0
        while True:
            episode_reward = reward_queue.get()
            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                running_reward = .99 * running_reward + .01 * episode_reward
                mean_rewards.append(running_reward)
            else:
                break
        for p in processes:
            p.join()
        torch.save(self.global_network.state_dict(), PATH)
        return episode_rewards, mean_rewards

    def test(self):
        screen = self.env.render(mode='rgb_array')
        im = Image.fromarray(screen)

        images = [im]

        state = self.env.reset()
        episode_reward = 0
        while True:
            action = self.global_network.get_action(torch.FloatTensor(state[None]))
            state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            screen = self.env.render(mode='rgb_array')
            images.append(Image.fromarray(screen))
            if done:
                print(f"episode_reward: {episode_reward}")
                break
        return images
