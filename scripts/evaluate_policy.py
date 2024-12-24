"""
DuckietownWorldEvaluator Evaluates an RLlib reinforcement learning agent using the same evaluator (duckietown-world),
which is used in Duckietown's official evaluator, DTS evaluate.
Duckietown World:
   takes trajectory data, which is recorded on one of it's "built in maps".
   https://github.com/duckietown/duckietown-world
   https://pypi.org/project/duckietown-world-daffy
   Supported version: 5.0.11 (designed for this, but may work with others)
DuckietownWorldEvaluator orchestrates the trajectory recording and evaluation for an RLlib agent.
Usage example:
   evaluator = DuckietownWorldEvaluator(config['env_config'])
   evaluator.evaluate(trainer, './EvaluationResults')

To adapt for a different agent implementation than RLlib, DuckietownWorldEvaluator should be subclassed and
   __init__ and _compute_action, (_collect_trajectory) should be overrided (modified)
Custom maps should be copied to the installation folder of Duckietown Worldlib .../duckietown_world/data/gd1/maps
   e.g. /home/username/miniconda3/envs/duckietownthesis/lib/python3.6/site-packages/duckietown_world/data/gd1/maps
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import os
import numpy as np
import json
import logging

from duckietown_world import SE2Transform
from duckietown_world.rules import evaluate_rules
from duckietown_world.rules.rule import EvaluatedMetric, make_timeseries
from duckietown_world.seqs.tsequence import SampledSequence
from duckietown_world.svg_drawing import draw_static
from duckietown_world.world_duckietown.duckiebot import DB18
from duckietown_world.world_duckietown.map_loading import load_map

from src.utils.video import VideoRecorder

import hydra
import torch
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

class DuckietownWorldEvaluator:
    """
    Evaluates agent using the same evaluator which is used in DTS evaluate.
    To adapt for a different agent implementation than RLlib, __init__ and _compute_action,
    (_collect_trajectory) should be modified.
    """


    def __init__(self, cfg):
        self.cfg = cfg
        self.map_name = cfg.env.map_name
        # Make testing env
        self.env = hydra.utils.instantiate(cfg.env, _recursive_=False)
        self.agent = hydra.utils.instantiate(cfg.agent, obs_shape=self.env.observation_space.shape,
                                             action_shape=self.env.action_space.shape)
        with Path(cfg.policy_path).open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

        self.work_dir = cfg.outdir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        # Set up evaluator
        # Creates an object 'duckiebot'
        self.ego_name = 'duckiebot'
        self.db = DB18()  # class that gives the appearance
        # load one of the maps
        self.dw = load_map(self.map_name)
        if cfg.record_video:
            self.video_recorder = VideoRecorder(
                Path(self.work_dir) if cfg.record_video else None)
        if cfg.record_agent_obs_video:
            self.agent_obs_video_recorder = VideoRecorder(
                Path(self.work_dir) if cfg.record_agent_obs_video else None, agent_obs=True)
        if cfg.record_transform_video:
            def transform(img):
                height, width, _ = img.shape
                img = img[:, width // 2 - height // 2:width // 2 + height // 2, :]
                return cv2.resize(img, (84, 84))
            
            self.transform_video_recorder = VideoRecorder(
                Path(self.work_dir) if cfg.record_transform_video else None, transform=transform)

    def evaluate(self, episodes=None):
        """
        Evaluates the agent on the map inicialised in __init__
        :param outdir: Directory for logged outputs (trajectory plots + numeric data)
        :param episodes: Number of evaluation episodes, if None, it is determined based on self.start_poses
        """
        outdir = self.work_dir
        if (outdir is not None) and (not os.path.exists(outdir)):
            os.makedirs(outdir)
        if episodes is None:
            episodes = len(self.start_poses.get(self.map_name, []))
        totals = {}
        for i in range(episodes):
            episode_path, episode_orientations, episode_timestamps = self._collect_trajectory(self.agent, i)
            logger.info("Episode {}/{} sampling completed".format(i+1, episodes))
            if len(episode_timestamps) <= 1:
                continue
            episode_path = np.stack(episode_path)
            episode_orientations = np.stack(episode_orientations)
            # Convert them to SampledSequences
            transforms_sequence = []
            for j in range(len(episode_path)):
                transforms_sequence.append(SE2Transform(episode_path[j], episode_orientations[j]))
            transforms_sequence = SampledSequence.from_iterator(enumerate(transforms_sequence))
            transforms_sequence.timestamps = episode_timestamps

            _outdir = outdir
            if outdir is not None and episodes > 1:
                _outdir = os.path.join(outdir, "Trajectory_{}".format(i+1))
            evaluated = self._eval_poses_sequence(transforms_sequence, outdir=_outdir)
            logger.info("Episode {}/{} plotting completed".format(i+1, episodes))
            totals = self._extract_total_episode_eval_metrics(evaluated, totals, display_outputs=True)

        # Calculate the median total metrics
        median_totals = {}
        mean_totals = {}
        stdev_totals = {}
        for key, value in totals.items():
            median_totals[key] = np.median(value)
            mean_totals[key] = np.mean(value)
            stdev_totals[key] = np.std(value)
        # Save results to file
        if outdir is not None:
            with open(os.path.join(outdir, "total_metrics.json"), "w") as json_file:
                json.dump({'median_totals': median_totals,
                        'mean_totals': mean_totals,
                        'stdev_totals': stdev_totals,
                        'episode_totals': totals}, json_file, indent=2)

        logger.info("\nMedian total metrics: \n {}".format(median_totals))
        logger.info("\nMean total metrics: \n {}".format(mean_totals))
        logger.info("\nStandard deviation of total metrics: \n {}".format(stdev_totals))

    def _collect_trajectory(self, agent, i):
        episode_path = []
        episode_orientations = []
        episode_timestamps = []
        obs = self.env.reset()
        if self.cfg.record_video:
            self.video_recorder.init(self.env, enabled=True)
        if self.cfg.record_agent_obs_video:
            self.agent_obs_video_recorder.init(self.env, enabled=True)
        if self.cfg.record_transform_video:
            self.transform_video_recorder.init(self.env, enabled=True)

        done = False
        while not done:
            action = self._compute_action(agent, obs)
            obs, reward, done, info = self.env.step(action)
            if self.cfg.record_video:
                self.video_recorder.record(self.env)
            if self.cfg.record_agent_obs_video:
                self.agent_obs_video_recorder.record(self.env)
            if self.cfg.record_transform_video:
                self.transform_video_recorder.record(self.env)
            cur_pos = np.array([self.env.cur_pos[0], self.env.grid_height * self.env.road_tile_size - self.env.cur_pos[2]]) 
            episode_path.append(cur_pos)
            episode_orientations.append(np.array(self.env.unwrapped.cur_angle))
            episode_timestamps.append(info['Simulator']['timestamp'])
        self.env.unwrapped.start_pose = None
        self.user_tile_start = None
        if self.cfg.record_video:
            self.video_recorder.save(f"episode_{i}.mp4")
        if self.cfg.record_agent_obs_video:
            self.agent_obs_video_recorder.save(f"episode_{i}_agent_obs.mp4")
        if self.cfg.record_transform_video:
            self.transform_video_recorder.save(f"episode_{i}_transform.mp4")
        return episode_path, episode_orientations, episode_timestamps

    def _compute_action(self, agent, obs):
        """
        This function should be modified for other agents!
        :param agent: Agent to be evaluated.
        :param obs: New observation
        :return: Action computed based on action
        """
        return agent.act(obs, 1, eval_mode=True)

    def _eval_poses_sequence(self, poses_sequence, outdir=None):
        """
        :param poses_sequence:
        :param outdir: If None evaluation outputs plots won't be saved
        :return:
        """
        # puts the object in the world with a certain "ground_truth" constraint
        self.dw.set_object(self.ego_name, self.db, ground_truth=poses_sequence)
        # Rule evaluation (do not touch)
        interval = SampledSequence.from_iterator(enumerate(poses_sequence.timestamps))
        evaluated = evaluate_rules(poses_sequence=poses_sequence,
                                   interval=interval, world=self.dw, ego_name=self.ego_name)
        if outdir is not None:
            timeseries = make_timeseries(evaluated)
            draw_static(self.dw, outdir, timeseries=timeseries)
        print(self.dw.get_drawing_children())
        self.dw.remove_object(self.ego_name)
        # self.dw.remove_object('visualization')
        return evaluated

    @staticmethod
    def _extract_total_episode_eval_metrics(evaluated, totals, display_outputs=False):
        episode_totals = {}
        for k, rer in evaluated.items():
            from duckietown_world.rules import RuleEvaluationResult
            assert isinstance(rer, RuleEvaluationResult)
            for km, evaluated_metric in rer.metrics.items():
                assert isinstance(evaluated_metric, EvaluatedMetric)
                episode_totals[k] = evaluated_metric.total
                if not (k in totals):
                    totals[k] = [evaluated_metric.total]
                else:
                    totals[k].append(evaluated_metric.total)
        if display_outputs:
            logger.info("\nEpisode total metrics: \n {}".format(episode_totals))

        return totals

@hydra.main(config_path="../cfgs", config_name="evaluate_policy")
def evaluate(cfg):
    evaluator = DuckietownWorldEvaluator(cfg)
    evaluator.evaluate(episodes=cfg.episodes)

if __name__ == "__main__":
    evaluate()