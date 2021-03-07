import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from utils.misc import compute_discounted_rewards

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Policy(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_units):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        return self.fc2(x)


def compute_loss(action_probs, discounted_rewards):
    log_action_probs = torch.log(action_probs)
    return -torch.sum(log_action_probs * discounted_rewards)


class Reinforce:
    def __init__(self, state_dim, n_actions, hidden_units, optim=optim.Adam, lr=1e-4):
        self.policy = Policy(state_dim, n_actions, hidden_units)
        self.policy.to(device)
        self.optim = optim(params=self.policy.parameters(), lr=lr)
        self.n_actions = n_actions

    def train(self, seed, env, max_episodes, gamma, max_episode_length):
        torch.manual_seed(seed)
        running_reward = 0
        episode_rewards, mean_rewards = [], []
        with tqdm.trange(0, max_episodes) as t:
            for episode in t:
                self.optim.zero_grad()
                rewards, action_probs = self._run_episode(env, max_episode_length)
                discounted_rewards = compute_discounted_rewards(rewards, gamma, device)
                loss = compute_loss(action_probs, discounted_rewards)
                loss.backward()
                self.optim.step()

                t.set_description(f"Episode {episode}")
                episode_reward = torch.sum(rewards)
                running_reward = .99 * running_reward + .01 * episode_reward

                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                mean_rewards.append(running_reward)
                episode_rewards.append(episode_reward)

        return episode_rewards, mean_rewards

    def _run_episode(self, env, max_episode_length):
        rewards, action_probs = [], []
        state = env.reset()
        for step in range(max_episode_length):
            state = torch.FloatTensor(state).to(device)
            action_logits = self.policy(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).cpu().detach().numpy()
            action_prob = distr.probs[action]
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            action_probs.append(action_prob)
            if done:
                break
        return torch.Tensor(rewards), torch.stack(action_probs)

    def test(self, seed, env):
        torch.manual_seed(seed)
        state = env.reset()
        while True:
            state = torch.FloatTensor(state).to(device)
            action_logits = self.policy(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).cpu().detach().numpy()
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
