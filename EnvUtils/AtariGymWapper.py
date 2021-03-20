import cv2
import gym
import numpy as np


# source: cnicholls RL with the A3C aagorithim

class AtariGymWrapper(gym.Wrapper):
    def __init__(self, env, skip_obs=4, num_frames=4, width=84, height=84):
        super(AtariGymWrapper, self).__init__(env)
        self.env = env
        self.skip_obs = skip_obs
        self.num_frames = num_frames
        self.width = width
        self.height = height

    # preprocess Atari Gym Obs according to Minh (2015)
    def _preprocess(self, obs, start_state=False):
        obs_gray_scaled = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
        state = cv2.resize(obs_gray_scaled, dsize=(self.width, self.height)) * (1.0 / 255.0)
        state = state.reshape(self.width, self.height, 1)
        if start_state or self.state is None:
            self.state = np.repeat(state, self.num_frames, axis=-1)
        else:
            self.state = np.append(state, self.state[:,:, :self.num_frames - 1], axis=-1)
        return self.state

    def reset(self):
        return self._preprocess(self.env.reset(), start_state=True)

    def step(self, action):
        accum_reward = 0
        for _ in range(self.skip_obs):
            state, reward, done, info = self.env.step(action)
            reward += reward
            if done:
                break
        return self._preprocess(state), accum_reward, done, info


if __name__ == '__main__':
    env = AtariGymWrapper(gym.make('Pong-v0'))
    state = env.reset()
    while True:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break
