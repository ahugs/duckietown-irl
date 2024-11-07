import gym 
import numpy as np
import hydra
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
        super().__init__(*args, **kwargs)
        
    def valid_pose(self, pos, angle):
        return super()._valid_pose(pos, angle)
                         
    def step(self, action):
        obs, _, done, info = super().step(action)
        return obs, 0, done, info
    
def make_duckietown(map_name, action_repeat, seed, image_size=84, episode_length=1000,
                    accept_start_angle_deg=4, wrappers=None, reward_wrappers=None):
    max_episode_steps = episode_length
    env_kwargs = {'map_name': map_name,
                  'camera_width': image_size,
                  'camera_height': image_size,
                  'max_steps': max_episode_steps,
                  'accept_start_angle_deg': accept_start_angle_deg}
    

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
