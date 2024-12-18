import gym 
import numpy as np
import hydra
import os
from collections.abc import Iterable
from gym_duckietown.utils import get_subdir_path
from gym_duckietown.simulator import Simulator
from gym_duckietown.exceptions import NotInLane
from src.envs.gym_wrappers import FrameStackWrapper, ChannelFirstWrapper, \
    DtRewardCollisionAvoidance, DtRewardVelocity, DtRewardPosAngle, NormalizeWrapper

gym.register(
    id='Duckietown-Simulator-v0',
    entry_point="src.envs.duckietown_env:DuckietownEnv"
)

class DuckietownEnv(Simulator):

    def __init__(self, *args, **kwargs):
        map_names = None
        if isinstance(kwargs["map_name"], Iterable) and not isinstance(kwargs["map_name"], str):
            map_names = kwargs["map_name"]
            kwargs['map_name'] = map_names[0]
        super().__init__(*args, **kwargs)

        if map_names is not None:
            self.randomize_maps_on_reset = True
            self.map_names = map_names
            self.reset()
            self.last_action = np.array([0, 0])
            self.wheelVels = np.array([0, 0])
        
    def valid_pose(self, pos, angle):
        return super()._valid_pose(pos, angle)
                         
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info
    
def make_duckietown(map_name, action_repeat, seed, image_size=84, episode_length=1000,
                    accept_start_angle_deg=4, wrappers=None, reward_wrappers=None,
                    full_transparency=False):
    max_episode_steps = episode_length
    env_kwargs = {'map_name': map_name,
                  'camera_width': image_size,
                  'camera_height': image_size,
                  'max_steps': max_episode_steps,
                  'accept_start_angle_deg': accept_start_angle_deg,
                  'full_transparency': full_transparency}
    

    # shorten episode length

    env = gym.make(
        id= f"Duckietown-Simulator-v0",
        seed=seed,
        frame_skip=action_repeat,
        **env_kwargs
    )
    if wrappers is not None:
        for wrapper in wrappers:
            print(wrapper)
            env = hydra.utils.instantiate(wrapper, env=env)
    if reward_wrappers is not None:
        for wrapper in reward_wrappers:
            env = hydra.utils.instantiate(wrapper, env=env)
    return env
