import torch


def compute_discounted_rewards(rewards, gamma, device, normalize_rewards=True):
    discounted_rewards = []
    gammas = torch.FloatTensor([pow(gamma, i) for i in range(len(rewards))])
    for step in range(len(rewards)):
        discounted_reward = torch.sum(rewards * gammas) if step == 0 \
            else torch.sum(rewards[:-step] * gammas[:-step])
        discounted_rewards.append(discounted_reward)
    discounted_rewards = torch.stack(discounted_rewards)

    if normalize_rewards:
        discounted_rewards = (discounted_rewards - torch.mean(discounted_rewards)) / \
                             torch.std(discounted_rewards)
    return discounted_rewards.to(device)
