import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from utils.misc import compute_discounted_rewards

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
huber_loss = nn.SmoothL1Loss(reduction='sum')


class Policy(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_units):
        super(Policy, self).__init__()
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


def compute_loss(action_probs, rewards, values):
    log_action_probs = torch.log(action_probs)
    advs = rewards - values
    return -torch.sum(log_action_probs * advs)


class ReinforceBaseline:
    def __init__(self, state_dim, n_actions, hidden_units, optim=optim.Adam, lr=1e-4):
        self.policy = Policy(state_dim, n_actions, hidden_units)
        self.policy.to(device)
        self.optim = optim(self.policy.parameters(), lr=lr)
        self.n_actions = n_actions

    def train(self, env, max_episodes, gamma, max_episode_length):
        running_reward = 0
        episode_rewards, mean_rewards = [], []
        with tqdm.trange(0, max_episodes) as t:
            for episode in t:
                self.optim.zero_grad()
                rewards, action_probs, values = self._run_episode(env, max_episode_length)
                values = torch.squeeze(values)
                discounted_rewards = compute_discounted_rewards(rewards, gamma, device, normalize_rewards=False)
                loss = torch.mean(
                    compute_loss(action_probs, discounted_rewards, values) + huber_loss(discounted_rewards, values))
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
        rewards, action_probs, state_values = [], [], []
        state = env.reset()
        for step in range(max_episode_length):
            state = torch.FloatTensor(state).to(device)
            action_logits, state_value = self.policy(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).cpu().detach().numpy()
            action_prob = distr.probs[action]
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            action_probs.append(action_prob)
            state_values.append(state_value)

            if done:
                break
        return torch.Tensor(rewards), torch.stack(action_probs), torch.stack(state_values)

    def test(self, env):
        state = env.reset()
        while True:
            state = torch.FloatTensor(state).to(device)
            action_logits, _ = self.policy(state)
            distr = torch.distributions.Categorical(logits=action_logits)
            action = torch.squeeze(distr.sample([1])).cpu().detach().numpy()
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
