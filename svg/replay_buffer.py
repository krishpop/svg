# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import numpy.random as npr
import torch
import os
import copy

from sortedcontainers import SortedSet

import pickle as pkl

from . import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self, num_envs, obs_shape, action_shape, capacity, device, normalize_obs
    ):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device

        self.pixels = len(obs_shape) > 1
        self.empty_data()

        self.done_idxs = [SortedSet() for i in range(num_envs)]
        self.global_idx = 0
        self.global_last_save = 0

        self.normalize_obs = normalize_obs

        if normalize_obs:
            assert not self.pixels
            self.welford = utils.Welford()

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del (
            d["obses"],
            d["next_obses"],
            d["actions"],
            d["rewards"],
            d["not_dones"],
            d["not_dones_no_max"],
        )
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # Manually need to re-load the transitions with load()
        self.empty_data()

    def empty_data(self):
        obs_dtype = np.float32 if not self.pixels else np.uint8
        n = self.num_envs
        obs_shape = self.obs_shape
        action_shape = self.action_shape
        capacity = self.capacity

        self.obses = np.empty((capacity, n, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, n, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, n, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, n), dtype=np.float32)
        self.not_dones = np.empty((capacity, n), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, n), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.payload = []
        self.done_idxs = None

    def __len__(self):
        return self.capacity if self.full else self.idx * self.num_envs

    def get_obs_stats(self):
        assert not self.pixels
        MIN_STD = 1e-1
        MAX_STD = 10
        mean = self.welford.mean()
        std = self.welford.std()
        std = np.clip(std, MIN_STD, MAX_STD)
        # print("mean_shape", mean.shape)
        # print("std_shape", std.shape)
        return mean, std

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        obs = utils.maybe_numpy(obs)
        action = utils.maybe_numpy(action)
        reward = utils.maybe_numpy(reward)
        next_obs = utils.maybe_numpy(next_obs)
        done = utils.maybe_numpy(done)
        done_no_max = utils.maybe_numpy(done_no_max)

        # For saving
        self.payload.append(
            (
                obs.copy(),
                next_obs.copy(),
                action.copy(),
                reward,
                1.0 - done,  # NOTE: unsure; used to be not done
                1.0 - done_no_max,  # NOTE: unsure; used to be not done_no_max
            )
        )

        if self.normalize_obs:
            self.welford.add_data(obs)

        if np.any(done):
            for i in done.nonzero()[0]:
                self.done_idxs[i.item()].add(self.idx)
        elif self.full:
            for i in done.nonzero()[0]:
                self.done_idxs[i.item()].discard(self.idx)

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        # NOTE: unsure; used to be not done
        np.copyto(self.not_dones[self.idx], 1.0 - done)
        # NOTE: unsure; used to be not done
        np.copyto(self.not_dones_no_max[self.idx], 1.0 - done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.global_idx += 1
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        m_batch_size = batch_size // self.num_envs
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=m_batch_size
        )

        obses = self.obses[idxs].reshape((-1, *self.obs_shape))
        actions = self.actions[idxs].reshape((-1, *self.action_shape))
        rewards = self.rewards[idxs].reshape((-1, 1))
        next_obses = self.next_obses[idxs].reshape((-1, *self.obs_shape))
        not_dones = self.not_dones[idxs].reshape((-1, 1))
        not_dones_no_max = self.not_dones_no_max[idxs].reshape((-1, 1))

        if self.normalize_obs:
            mu, sigma = self.get_obs_stats()
            obses = (obses - mu) / sigma
            next_obses = (next_obses - mu) / sigma

        obses = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_obses = torch.as_tensor(
            next_obses, dtype=torch.float32, device=self.device
        )
        not_dones = torch.as_tensor(not_dones, device=self.device)
        not_dones_no_max = torch.as_tensor(not_dones_no_max, device=self.device)

        assert obses.shape == (batch_size, *self.obs_shape), obses.shape
        assert actions.shape == (batch_size, *self.action_shape), actions.shape
        assert rewards.shape == (batch_size, 1), rewards.shape
        # assert not_dones == (batch_size, 1), not_dones.shape
        # assert not_dones_no_max == (batch_size, 1), not_dones_no_max.shape

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def sample_multistep(self, batch_size, T):
        # print("batch_size", batch_size)
        # print("idx", self.idx)
        # print("full?", self.full)
        assert batch_size < self.idx * self.num_envs or self.full

        m_batch_size = batch_size // self.num_envs
        assert (batch_size / self.num_envs).is_integer()
        assert m_batch_size > 0

        last_idx = self.capacity if self.full else self.idx
        last_idx -= T

        o, a, r = [], [], []

        for n in range(self.num_envs):
            # raw here means the "coalesced" indices that map to valid
            # indicies that are more than T steps away from a done
            done_idxs_sorted = np.array(list(self.done_idxs[n]) + [last_idx])
            n_done = len(done_idxs_sorted)
            done_idxs_raw = done_idxs_sorted - np.arange(1, n_done + 1) * T

            # print("n=", n)
            # print("last_idx", last_idx)
            # print("a", last_idx - (T + 1) * n_done)
            # print("n_done", n_done)
            # print("done_idxs_sorted", done_idxs_sorted)
            samples_raw = npr.choice(
                last_idx - (T + 1) * n_done,
                size=m_batch_size,
                replace=True,
            )
            samples_raw = sorted(samples_raw)
            js = np.searchsorted(done_idxs_raw, samples_raw)
            offsets = done_idxs_raw[js] - samples_raw + T
            start_idxs = done_idxs_sorted[js] - offsets

            obses, actions, rewards = [], [], []

            for t in range(T):
                obses.append(self.obses[start_idxs + t, n])
                actions.append(self.actions[start_idxs + t, n])
                rewards.append(self.rewards[start_idxs + t, n])
                assert np.all(self.not_dones[start_idxs + t, n])

            obses = np.stack(obses)
            actions = np.stack(actions)
            rewards = np.stack(rewards)  # .squeeze(2)

            if self.normalize_obs:
                mu, sigma = self.get_obs_stats()
                obses = (obses - mu) / sigma

            o.append(torch.as_tensor(obses, device=self.device).float())
            a.append(torch.as_tensor(actions, device=self.device))
            r.append(torch.as_tensor(rewards, device=self.device))
            # print(torch.as_tensor(obses, device=self.device).shape)

        obses = torch.concat(o, dim=1)
        actions = torch.concat(a, dim=1)
        rewards = torch.concat(r, dim=1)

        assert obses.shape == (T, batch_size, *self.obs_shape), obses.shape
        assert actions.shape == (T, batch_size, *self.action_shape), actions.shape
        assert rewards.shape == (T, batch_size), rewards.shape

        return obses, actions, rewards

    def save_data(self, save_dir):
        if self.global_idx == self.global_last_save:
            return
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(
            save_dir, f"{self.global_last_save:08d}_{self.global_idx:08d}.pt"
        )

        payload = list(zip(*self.payload))
        payload = [np.vstack(x) for x in payload]
        self.global_last_save = self.global_idx
        torch.save(payload, path)
        self.payload = []

    def load_data(self, save_dir):
        def parse_chunk(chunk):
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            return (start, end)

        self.idx = 0

        chunks = os.listdir(save_dir)
        chunks = filter(lambda fname: "stats" not in fname, chunks)
        chunks = sorted(chunks, key=lambda x: int(x.split("_")[0]))

        self.full = self.global_idx > self.capacity
        global_beginning = self.global_idx - self.capacity if self.full else 0

        for chunk in chunks:
            global_start, global_end = parse_chunk(chunk)
            if global_start >= self.global_idx:
                continue
            start = global_start - global_beginning
            end = global_end - global_beginning
            if end <= 0:
                continue

            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            if start < 0:
                payload = [x[-start:] for x in payload]
                start = 0
            assert self.idx == start

            obses = payload[0]
            next_obses = payload[1]

            self.obses[start:end] = obses
            self.next_obses[start:end] = next_obses
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.not_dones_no_max[start:end] = payload[5]
            self.idx = end

        self.last_save = self.idx

        if self.full:
            assert self.idx == self.capacity
            self.idx = 0

        last_idx = self.capacity if self.full else self.idx
        self.done_idxs = SortedSet(np.where(1.0 - self.not_dones[:last_idx])[0])
