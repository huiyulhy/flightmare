#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import gym 
import numpy as np 
import torch 

#
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.ppo.ppo import PPO

#
import rpg_baselines.common.util as U
from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1

# check that system is using cuda 
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

print(use_cuda)
print(device)

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.goal = np.zeros((3,1))

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.locals['new_obs'][0, 0:3] = self.locals['new_obs'][0, 0:3] - np.reshape(self.goal, self.locals['new_obs'][0, 0:3].shape)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print("rollout ended: callback entered")

    def set_goal(self, goal): 
        self.goal = goal