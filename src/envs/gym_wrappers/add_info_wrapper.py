from gym import Wrapper

class AddInfoWrapper(Wrapper):

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}