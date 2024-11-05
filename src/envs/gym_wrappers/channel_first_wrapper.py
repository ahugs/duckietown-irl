from gym import Wrapper
from gym.spaces import Box

class ChannelFirstWrapper(Wrapper):   
    def __init__(self, env):       
        super().__init__(env, lambda obs: obs.transpose(2, 0, 1))
        obs_shape = env.observation_space.shape
        low = self.observation_space.low.reshape(obs_shape[2], obs_shape[0], obs_shape[1])
        high = self.observation_space.high.reshape(obs_shape[2], obs_shape[0], obs_shape[1])
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )   
