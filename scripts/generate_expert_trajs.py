import os
import numpy as np
import hydra
# parameters for the pure pursuit controller
from gym_duckietown.simulator import get_right_vec
import wandb
from omegaconf import OmegaConf
from pathlib import Path

from src.utils.video import VideoRecorder

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.3
GAIN = 10
FOLLOWING_DISTANCE = 0.3
TRIM = 0.0
RADIUS = 0.0318
K = 27.0
LIMIT = 1.0


class PurePursuitExpert:
    def __init__(
        self,
        env,
        ref_velocity=REF_VELOCITY,
        position_threshold=POSITION_THRESHOLD,
        following_distance=FOLLOWING_DISTANCE,
        max_iterations=1000,
    ):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(get_right_vec(self.env.cur_angle), point_vec)
        steering = GAIN * -dot

        return self.convert_to_wheel_vels(self.ref_velocity, steering)
    
    def convert_to_wheel_vels(self, vel, angle):

        # Distance between the wheels
        baseline = self.env.unwrapped.wheel_dist

        # adjusting k by gain and trim
        k_r_inv = 1 / K
        k_l_inv = 1 / K

        omega_r = (vel + 0.5 * angle * baseline) / RADIUS
        omega_l = (vel - 0.5 * angle * baseline) / RADIUS

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, LIMIT), -LIMIT)
        u_l_limited = max(min(u_l, LIMIT), -LIMIT)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

@hydra.main(config_path="../cfgs", config_name="expert_trajs")
def generate_trajectories(cfg):
    # Initialize the environment
    print(Path(os.getcwd()).parent / Path('temp'))

    wandb_run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True))
    env = hydra.utils.call(cfg.env, _recursive_=False)    
    video_recorder = VideoRecorder(Path(os.getcwd()).parent, wrun=wandb_run)
    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)



    # let's collect our samples
    for episode in range(0, cfg.num_trajs):
        print("Starting episode", episode)
        env.reset()
        video_recorder.init(env, enabled=True)
        done = False
        ep_reward = 0
        ep_collision = 0
        ep_not_in_lane = 0
        ep_invalid_pose = 0
        observations = []
        actions = []
        rewards = []
        dones = []
        while not done:
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            ep_reward += reward
            ep_collision += info['collision']
            ep_not_in_lane += info['not_in_lane']
            ep_invalid_pose += info['invalid_pose']
            video_recorder.record(env)

        np.savez(f'{cfg.save_dir}/episode_{episode}_{len(rewards)}.npz', 
                 observation=np.array(observations), action=np.array(actions), 
                          reward=np.array(rewards), done=np.array(dones))
        video_recorder.save(f"episode_{episode}.mp4")
        wandb_run.log({'episode_reward': ep_reward,
                       'invalid_pose': ep_invalid_pose,
                       'not_in_lane': ep_not_in_lane,
                       'collision': ep_collision})


if __name__ == "__main__":
    generate_trajectories()