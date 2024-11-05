import numpy as np

from gym.wrappers.frame_stack import FrameStack
from gym.spaces import Box

class FrameStackWrapper(FrameStack):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        super().__init__(env, num_frames)
        obs_shape = self.observation_space.shape
        low = self.observation_space.low.reshape(obs_shape[0]*obs_shape[1], obs_shape[2], obs_shape[3])
        high = self.observation_space.high.reshape(obs_shape[0]*obs_shape[1], obs_shape[2], obs_shape[3])
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
    def reset(self):
        obs = super().reset()
        return obs.__array__().reshape(self.observation_space.shape)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs.__array__().reshape(self.observation_space.shape), reward, done, info
