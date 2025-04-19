import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
    Collector
)

from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy


class PackCollector(Collector):
    """Collector enables the policy to interact with different types of envs with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive the keys "obs_next", "rew",
    "terminated", "truncated, "info", "policy" and "env_id" in a normal env step.
    Alternatively, it may also accept the keys "obs_next", "rew", "done", "info",
    "policy" and "env_id".
    It returns either a dict or a :class:`~tianshou.data.Batch` with the modified
    keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer that was passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        exploration_noise: bool = False,
    ) -> None:
        super().__init__(policy=policy, env=env, buffer=buffer, exploration_noise=exploration_noise)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False. 
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
        """
        """
        Collect a specified number of steps or episodes, then return stats dict
        with exactly the same keys as your original version.
        """
        # 1) Use the base collect to get a CollectStats dataclass
        stats_obj = super().collect(
            n_step=n_step,
            n_episode=n_episode,
            random=random,
            render=render,
            reset_before_collect=True,
            gym_reset_kwargs=gym_reset_kwargs,
        )

        # 2) Extract total steps from the stats object
        total_steps = getattr(stats_obj, "n_collected_steps", 0)

        # 3) Slice buffer and compute extra metrics
        batch = self.buffer[: total_steps]
        ratio_vals = np.array(batch.info.get("ratio", []))
        counter_vals = np.array(batch.info.get("counter", []))
        ratio_mean = float(ratio_vals.mean()) if ratio_vals.size else 0.0
        ratio_std = float(ratio_vals.std()) if ratio_vals.size else 0.0
        num_mean = float(counter_vals.mean()) if counter_vals.size else 0.0
        num_std = float(counter_vals.std()) if counter_vals.size else 0.0

        # 4) Attach new fields directly to stats_obj
        stats_obj.ratios = ratio_vals
        stats_obj.ratio = ratio_mean
        stats_obj.ratio_std = ratio_std
        stats_obj.nums = counter_vals
        stats_obj.num = num_mean
        stats_obj.num_std = num_std

        return stats_obj

