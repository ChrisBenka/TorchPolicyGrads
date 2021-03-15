import gym
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from SharedOptim import SharedAdam

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: list, n_actions: int):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], 24)
        self.fc2 = nn.Linear(state_dim[0], 48)
        self.fc3 = nn.Linear(24, n_actions)
        self.fc4 = nn.Linear(48, 1)

    def forward(self, inputs: torch.Tensor):
        action_hidden = F.hardtanh_(self.fc1(inputs))
        action_logits = self.fc3(action_hidden)
        state_hidden = F.hardtanh_(self.fc2(inputs))
        state_value = self.fc4(state_hidden)
        return action_logits, state_value

    def update_from_model(self, model: nn.Module):
        self.load_state_dict(model.state_dict())

    def get_action(self, obs):
        self.eval()
        action_logits, _ = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        distr = Categorical(probs=probs)
        action = distr.sample([1]).numpy()[0]
        return action

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor, td_targets: torch.Tensor):
        self.train()
        action_logits, state_values = self.forward(states)
        advs = td_targets - state_values
        probs = F.softmax(action_logits, dim=-1)
        distr = Categorical(probs=probs)
        critic_loss = advs ** 2
        actor_loss = -(distr.log_prob(actions) * advs.detach().squeeze())
        return (critic_loss + actor_loss).mean()

class Worker:
    def __init__(self, rank: int, seed: int, env: gym.Env, gamma: float, update_interval: int):
        self.rank = rank
        self.env = env
        self.env.seed(rank + seed)
        state_dim = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        self.gamma = gamma
        self.update_interval = update_interval
        self.t_actor_critic = ActorCriticNet(state_dim=state_dim, n_actions=n_actions)

    def work(self, g_actor_critic: nn.Module, g_optim: SharedAdam, global_episodes: mp.Value, reward_queue: mp.Queue,
             mean_reward: mp.Value, max_episodes: int, max_steps: int, env_target: int):
        print(f"starting {self.rank}")
        t_step_count = 0
        solved = False
        while True:
            self.t_actor_critic.update_from_model(g_actor_critic)
            episode_reward = 0
            state = self.env.reset()
            states, actions, rewards = [], [], []
            for _ in range(max_steps):
                action = self.t_actor_critic.get_action(torch.FloatTensor(state))
                next_s, reward, done, _ = self.env.step(action)

                if self.rank == 0:
                    self.env.render()

                episode_reward += reward
                t_step_count += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if done or t_step_count % self.update_interval == 0 and global_episodes.value <= max_episodes:
                    if done:
                        state = self.env.reset()
                        with mean_reward.get_lock():
                            mean_reward.value = mean_reward.value * .99 + episode_reward * .01
                            print(
                                f"worker:{self.rank} running_reward:{int(mean_reward.value)}")
                            if mean_reward.value >= env_target:
                                solved = True
                        R = 0
                    else:
                        R = self.t_actor_critic(torch.FloatTensor(state))[1].detach().numpy()[0]
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

            with global_episodes.get_lock():
                global_episodes.value += 1
                if global_episodes.value > max_episodes:
                    break
                else:
                    reward_queue.put(episode_reward)
                if solved:
                    print(f"Solved at {global_episodes.value}")
                    break

        reward_queue.put(None)
        print(f" exiting {self.rank}")


class A3C:
    def __init__(self, env_name, n_workers, lr):
        self.env_name = env_name
        self.n_workers = n_workers
        self.lr = lr
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.global_network = ActorCriticNet(state_dim=self.state_dim, n_actions=self.n_actions)
        self.global_network.share_memory()
        self.global_optim = SharedAdam(params=self.global_network.parameters(), lr=lr)

    def train(self, seed, max_episodes, gamma, max_steps, env_target, update_interval):
        global_episodes, reward_queue, mean_reward = mp.Value('i', 0), mp.Queue(), mp.Value('d', 0)
        workers, processes = [], []
        torch.manual_seed(seed)
        self.env.seed(seed)

        for rank in range(self.n_workers):
            worker = Worker(rank, seed, gym.make(self.env_name), gamma, update_interval)
            p = mp.Process(target=worker.work,
                           args=(self.global_network, self.global_optim, global_episodes, reward_queue,
                                 mean_reward, max_episodes, max_steps, env_target))
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
        return episode_rewards, mean_rewards

    def test(self):
        state = self.env.reset()
        episode_reward = 0
        while True:
            action = self.global_network.get_action(torch.FloatTensor(state))
            state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.env.render()
            if done:
                print(f"episode_reward: {episode_reward}")
                break
