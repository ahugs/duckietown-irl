import os
import numpy as np
import hydra
# parameters for the pure pursuit controller
from gym_duckietown.simulator import get_right_vec
import wandb
from omegaconf import OmegaConf
import torch
from pathlib import Path

from src.utils.video import VideoRecorder

@hydra.main(config_path="../cfgs", config_name="evaluate_policy")
def generate_trajectories(cfg):
    # Initialize the environment
    print(Path(os.getcwd()).parent / Path('temp'))

    wandb_run = wandb.init(config=OmegaConf.to_container(cfg, resolve=True))
    env = hydra.utils.call(cfg.env, _recursive_=False)    
    video_recorder = VideoRecorder(Path(os.getcwd()).parent, wrun=wandb_run)
    # Create an imperfect demonstrator
    policy = hydra.utils.instantiate(cfg.agent, obs_shape=env.observation_space.shape, 
                                     action_shape=env.action_space.shape)
    with cfg.snapshot.open('rb') as f:
        payload = torch.load(f)
    for k, v in payload.items():
        policy.__dict__[k] = v


    # let's collect our samples
    for episode in range(0, cfg.num_trajs):
        print("Starting episode", episode)
        observation = env.reset()
        video_recorder.init(env, enabled=True)
        done = False
        ep_reward = 0
        ep_collision = 0
        ep_not_in_lane = 0
        ep_invalid_pose = 0
        ep_dist_from_lane = 0
        ep_angle_from_lane = 0
        ep_speed = 0

        observations = []
        actions = []
        rewards = []
        dones = []
        while not done:
            # use our 'expert' to predict the next action.
            action = policy.act(observation, step=1, eval_mode=True)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            ep_reward += reward
            ep_collision += info['collision']
            ep_not_in_lane += info['not_in_lane']
            ep_invalid_pose += info['invalid_pose']
            ep_dist_from_lane += info['dist_from_lane']
            ep_angle_from_lane += info['angle_from_lane']
            ep_speed += info['speed']
            video_recorder.record(env)

        if cfg.save_dir is not None:
            np.savez(f'{cfg.save_dir}/episode_{episode}_{len(rewards)}.npz', 
                    observation=np.array(observations), action=np.array(actions), 
                            reward=np.array(rewards), done=np.array(dones))
        video_recorder.save(f"episode_{episode}.mp4")
        wandb_run.log({'episode_reward': ep_reward,
                       'invalid_pose': ep_invalid_pose,
                       'not_in_lane': ep_not_in_lane,
                       'collision': ep_collision,
                       'dist_from_lane': ep_dist_from_lane,
                       'angle_from_lane': ep_angle_from_lane,
                       'speed': ep_speed})


if __name__ == "__main__":
    generate_trajectories()