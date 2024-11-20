# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import sys
# import os
# sys.stderr.write(f"DISPLAY: {os.environ['DISPLAY']}\n")

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# import subprocess
# xvfb_subproc = subprocess.Popen(("Xvfb", os.environ["DISPLAY"], '-screen', '0', '1024x768x24', '-ac',  
#                                     '+extension',  'GLX',  '+render', '-noreset'))
import os

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from dataclasses import dataclass

import src.utils.utils as utils
from src.utils.logger import Logger
from src.utils.replay_buffer import ReplayBufferStorage, make_replay_loader
from src.utils.video import TrainVideoRecorder, VideoRecorder
from src.envs.duckietown_env import make_duckietown
from src.envs import specs

torch.backends.cudnn.benchmark = True


def make_agent(obs_shape, action_shape, cfg):
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_space.shape,
                                self.train_env.action_space.shape,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        wandb.tensorboard.patch(root_logdir=f"{self.work_dir}")
        
        self.wandb_run = wandb.init(config=OmegaConf.to_container(self.cfg, resolve=True))

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        self.train_env = hydra.utils.call(self.cfg.env, _recursive_=False)

        self.eval_env = hydra.utils.call(self.cfg.env, _recursive_=False)

        # create replay buffer
        data_specs = (specs.Array(self.train_env.observation_space.shape, np.uint8, 'observation'),
                      specs.Array(self.train_env.action_space.shape, np.float32, 'action'),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.bool, 'done'))
        print(data_specs)

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None, wrun=self.wandb_run)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.env.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            obs = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            done = False
            while not done:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs,
                                            self.global_step,
                                            eval_mode=True)
                obs, reward, done, _ = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.env.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.env.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.env.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.env.action_repeat)

        episode_step, episode_reward = 0, 0
        obs = self.train_env.reset()
        self.replay_storage.add(observation=obs, 
                                action=np.array([np.nan]*self.train_env.action_space.shape[0], dtype=np.float32), 
                                reward=np.array([0], dtype=np.float32), done=np.array([False]))
        self.train_video_recorder.init(obs)
        metrics = None
        done = False
        while train_until_step(self.global_step):
            if done:
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.env.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                obs = self.train_env.reset()
                self.replay_storage.add(observation=obs, 
                                        action=np.array([np.nan]*self.train_env.action_space.shape[0], dtype=np.float32), 
                                reward=np.array([0], dtype=np.float32), done=np.array([False]))
                self.train_video_recorder.init(obs)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs,
                                        self.global_step,
                                        eval_mode=False)
                # print(action)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            obs, reward, done, info = self.train_env.step(action)
            episode_reward += reward
            self.replay_storage.add(observation=obs, reward=np.array([reward], dtype=np.float32), 
                                    action=action, done=np.array([done]))
            self.train_video_recorder.record(obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()