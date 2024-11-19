import numpy as np 

from gym import Wrapper, Env
from gym_duckietown.simulator import NotInLane


class CostInfoWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['invalid_pose'] = not self.valid_pose(self.cur_pos, self.cur_angle)
        try:
            lp = self.get_lane_pos2(self.cur_pos, self.cur_angle)
            info['not_in_lane'] = False
            info['dist_from_lane'] = lp.dist
            info['angle_from_lane'] = lp.dot_dir
        except NotInLane:
            info['not_in_lane'] = True
            info['dist_from_lane'] = np.nan
            info['angle_from_lane'] = np.nan
        info['collision'] = self.proximity_penalty2(self.cur_pos, self.cur_angle) > 0
        return obs, reward, done, info