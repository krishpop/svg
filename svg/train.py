#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

# import isaacgym
import torch
import copy
import os
import sys
import shutil
import time
import pickle as pkl
from omegaconf import OmegaConf
import yaml


from setproctitle import setproctitle

setproctitle("svg")

import hydra

# import hydra_plugins
import wandb

from svg import sweeper

from svg.video import VideoRecorder
from svg import agent, utils, temp, dx, actor, critic
from svg.logger import Logger
from svg.replay_buffer import ReplayBuffer

# try:
#     if os.isatty(sys.stdout.fileno()):
#         from IPython.core import ultratb

#         sys.excepthook = ultratb.FormattedTB(
#             mode="Verbose", color_scheme="Linux", call_pdb=1
#         )
# except:
#     pass
import torch
import torch.nn as nn
import numpy as np


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_freq,
            agent="sac_svg",
        )

        # utils.set_seed_everywhere(cfg.seed)x
        self.device = torch.device(cfg.device)
        # self.env = utils.make_norm_env(cfg)
        self.env = hydra.utils.instantiate(cfg.env)
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.score_keys = cfg.score_keys

        cfg.obs_dim = int(self.env.observation_space.shape[0])
        cfg.action_dim = self.env.action_space.shape[0]
        cfg.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        if isinstance(cfg.replay_buffer_capacity, str):
            cfg.replay_buffer_capacity = int(eval(cfg.replay_buffer_capacity))

        self.replay_buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
            normalize_obs=cfg.normalize_obs,
        )
        self.replay_dir = os.path.join(self.work_dir, "replay")

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.step = 0
        self.steps_since_eval = 0
        self.steps_since_save = 0
        self.best_eval_rew = None

    def evaluate(self):
        episode_rewards = []
        for episode in range(self.cfg.num_eval_episodes):
            # if self.cfg.fixed_eval:
                # self.env.set_seed(episode)
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs_norm = (obs - mu) / sigma
                        action = self.agent.act(obs_norm, sample=False)
                    else:
                        action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            for key in self.score_keys:
                self.logger.log(f"eval/scores/{key}/step", info[key], self.step)
                self.logger.log(f"eval/scores/{key}/time", info[key], self.step)
                self.logger.log(f"eval/scores/{key}/iter", info[key], self.step)

            self.video_recorder.save(f"{self.step}.mp4")
            self.logger.log("eval/episode_reward", episode_reward, self.step)
            self.logger.log("rewards/step", episode_reward, self.step)
        # if self.cfg.fixed_eval:
            # self.env.set_seed(None)
        self.logger.dump(self.step)
        return np.mean(episode_rewards)

    # @profile
    def run(self):
        assert self.episode_reward == 0.0
        assert self.episode_step == 0
        self.agent.reset()
        obs = self.env.reset()
        done = False

        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                # log training metric
                if self.step > 0:
                    self.logger.log(
                        "train/episode_reward", self.episode_reward, self.step
                    )
                    time_elapsed = time.time() - start_time
                    self.logger.log("train/duration", time_elapsed, self.step)
                    self.logger.log("train/episode", self.episode, self.step)
                    for key in self.score_keys:
                        if key in info:
                            self.logger.log(
                                f"train/scores/{key}/step", info[key], self.step
                            )
                            self.logger.log(
                                f"train/scores/{key}/iter", info[key], self.step
                            )
                            self.logger.log(
                                f"train/scores/{key}/time", info[key], time_elapsed
                            )

                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps)
                    )

                # eval if necessary
                if self.steps_since_eval >= self.cfg.eval_freq:
                    self.logger.log("eval/episode", self.episode, self.step)
                    eval_rew = self.evaluate()
                    self.steps_since_eval = 0

                    if self.best_eval_rew is None or eval_rew > self.best_eval_rew:
                        self.save(tag="best")
                        self.best_eval_rew = eval_rew

                    self.replay_buffer.save_data(self.replay_dir)
                    self.save(tag="latest")

                if (
                    self.step > 0
                    and self.cfg.save_freq
                    and self.steps_since_save >= self.cfg.save_freq
                ):
                    tag = str(self.step).zfill(self.cfg.save_zfill)
                    self.save(tag=tag)
                    self.steps_since_save = 0

                # if self.cfg.num_initial_states is not None:
                    # self.env.set_seed(self.episode % self.cfg.num_initial_states)
                obs = self.env.reset()
                self.agent.reset()
                done = False
                self.episode_reward = 0
                self.episode_step = 0
                self.episode += 1

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                # TODO make this scalable for single and vectorized environments
                action = []
                for _ in range(self.env.num_envs):
                    action.append(self.env.action_space.sample())
                action = np.stack(action)
            else:
                with utils.eval_mode(self.agent):
                    if self.cfg.normalize_obs:
                        mu, sigma = self.replay_buffer.get_obs_stats()
                        obs = utils.maybe_numpy(obs)
                        obs_norm = (obs - mu) / sigma
                        action = self.agent.act(obs_norm, sample=True)
                    else:
                        action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps - 1:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            action = torch.tensor(action, device=self.device)
            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done_float = done.float()
            done_no_max = (
                done_float
                if self.episode_step + 1 < self.cfg.max_episode_steps
                else 0.0
            )
            self.episode_reward += reward

            self.replay_buffer.add(
                obs, action, reward, next_obs, done_float, done_no_max
            )

            obs = next_obs
            self.episode_step += 1
            self.step += 1
            self.steps_since_eval += 1
            self.steps_since_save += 1

        if self.steps_since_eval > 1:
            self.logger.log("eval/episode", self.episode, self.step)
            self.evaluate()

        if self.cfg.delete_replay_at_end:
            shutil.rmtree(self.replay_dir)

    # @profile
    def run_epochs(self):
        # assert not self.done
        assert self.episode_reward == 0.0
        assert self.episode_step == 0
        self.agent.reset()
        obs = self.env.reset()

        episode_length_meter = AverageMeter(1, 100).to(self.device)
        episode_reward_meter = AverageMeter(1, 100).to(self.device)

        self.episode_reward = torch.zeros((self.env.num_envs,))
        self.episode_step = torch.zeros((self.env.num_envs,))

        epoch = 0
        while self.step < self.cfg.num_train_steps:
            # do a steps_num long rollout
            start_time = time.time()
            for _ in range(self.cfg.steps_num):
                # sample action for data collection
                if self.step < self.cfg.num_seed_steps:
                    # print("seed steps", self.cfg.num_seed_steps)
                    # print(self.step)
                    action = []
                    for _ in range(self.env.num_envs):
                        action.append(self.env.action_space.sample())
                    action = np.stack(action)
                else:
                    with utils.eval_mode(self.agent):
                        if self.cfg.normalize_obs:
                            mu, sigma = self.replay_buffer.get_obs_stats()
                            obs = utils.maybe_numpy(obs)
                            obs_norm = (obs - mu) / sigma
                            action = self.agent.act(obs_norm, sample=True)
                        else:
                            action = self.agent.act(obs, sample=True)

                action = torch.tensor(action, device=self.device)
                # print(self.step)
                # print(action.shape)
                # print(action)
                next_obs, reward, done, info = self.env.step(action)

                # allow infinite bootstrap
                done_float = done.float()
                done_no_max = utils.maybe_numpy(info["termination"])
                self.episode_reward += utils.maybe_numpy(reward)

                self.replay_buffer.add(
                    obs, action, reward, next_obs, done_float, done_no_max
                )

                obs = next_obs
                self.episode_step += 1
                self.step += self.env.num_envs
                epoch += 1

                # log all done environments
                dones = done.nonzero(as_tuple=False).squeeze(-1)
                episode_length_meter.update(self.episode_step[dones])
                self.episode_step[dones] = 0.0
                episode_reward_meter.update(self.episode_reward[dones])
                self.episode_reward[dones] = 0.0

            # update at the end of the rollout
            # TODO need to scale up num updates to account for num_steps
            if self.step > self.cfg.agent.step_batch_size:
                self.agent.update(self.replay_buffer, self.logger, self.step)

                end_time = time.time()

                fps = self.cfg.steps_num * self.env.num_envs / (end_time - start_time)

                r = episode_reward_meter.get_mean()
                l = episode_length_meter.get_mean()
                print(
                    f"{self.step}/{self.cfg.num_train_steps}, ep_reward: {r:.2f}, ep_len: {l:.2f}, fps: {fps:.2f}, dx_loss: {self.agent.rolling_dx_loss:.2f}, policy_loss: {self.agent.policy_loss:.2f}, critic_loss: {self.agent.critic_loss:.2f}"
                )

                # log metrics for envs that are done
                self.logger.log("reward", r, self.step)
                self.logger.log("episode_lengths", l, self.step)
                self.logger.log("fps", fps, self.step)
                self.logger.log("dx_loss", self.agent.rolling_dx_loss, self.step)

                # save checkpoints
                if self.step > 0 and epoch % self.cfg.save_freq == 0:
                    tag = str(self.step).zfill(self.cfg.save_zfill)
                    self.save(tag=tag)

        if self.cfg.delete_replay_at_end:
            shutil.rmtree(self.replay_dir)

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load(work_dir, tag="latest"):
        path = os.path.join(work_dir, f"{tag}.pkl")
        with open(path, "rb") as f:
            return pkl.load(f)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d["logger"], d["env"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        # override work_dir
        self.work_dir = os.getcwd()
        self.logger = Logger(
            self.work_dir,
            save_tb=self.cfg.log_save_tb,
            log_frequency=self.cfg.log_freq,
            agent="sac_svg",
        )
        self.env = utils.make_norm_env(self.cfg)
        if "max_episode_steps" in self.cfg and self.cfg.max_episode_steps is not None:
            self.env._max_episode_steps = self.cfg.max_episode_steps
        self.episode_step = 0
        self.episode_reward = 0
        done = False

        if os.path.exists(self.replay_dir):
            self.replay_buffer.load_data(self.replay_dir)


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg):
    # this needs to be done for successful pickle
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    from train import Workspace as W

    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        cfg_yaml = OmegaConf.to_yaml(cfg)
        params = yaml.safe_load(cfg_yaml)
        wandb.init(
            project="svg",
            config=params,
            entity="krshna",
            sync_tensorboard=True,
            resume="allow",
        )
        workspace = W(cfg)

    workspace.run()


if __name__ == "__main__":
    main()
