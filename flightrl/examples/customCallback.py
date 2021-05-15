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
from rpg_baselines.common.high_level_planner import HighLevelPlanner
from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1

# check that system is using cuda 
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

print(use_cuda)
print(device)

class errorCallback(BaseCallback):
    """
    A error callback class that derives from ``BaseCallback``.
    This class of error callbacks are designed to be used when the 
    training is performed on error instead of actual position 

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(errorCallback, self).__init__(verbose)
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
        self.high_level_planner = HighLevelPlanner(num_runs= 1,
                                    num_goals_per_run = 1,
                                    world_box = [-15, -15, 0, 15, 15, 8], 
                                    min_distance_between_consecutive_goals = 10,
                                    switch_goal_distance = 3)
        self.possible_goals = np.array([[1.0, 0.0, 4.0], 
                                [0.0, 0.0, 3.0], 
                                [0.0, 1.0, 4.0],
                                [0.0, -1.0, 4.0], 
                                [1.0, 0.0, 4.0], 
                                [0.0, 0.0, 5.0]],
                                dtype=np.float32)

    def processObs(self, obs, current_goal) -> None: 
        """
        This method is the preprocessing pipeline of the observation
        before passing observations into the PPO agent for training
        """
        new_obs = obs.copy()
        new_obs[:,0:3] = new_obs[:, 0:3] - np.reshape(current_goal, new_obs[:, 0:3].shape)
        new_obs[:,12:] = np.zeros(new_obs[:,12:].shape)
        return new_obs        

    def _on_rollout_start(self) -> None:  
        """
        This method is called before the first rollout starts 
        """
        #randint = np.random.randint(0, self.possible_goals.shape[0])
        #self.goal = self.possible_goals[randint]
        self.goal = self.high_level_planner.draw_random_position()
        self.goal = np.multiply(self.goal, [0.2, 0.2, 0.5])
        self.goal = self.goal + np.array([0.0, 0.0, 2.0])
        self.goal = np.array(self.goal, dtype=np.float32)
        self.training_env.set_goal(self.goal)
        print("current goal " + str(self.goal))
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.locals['new_obs'] = self.processObs(self.locals['new_obs'], self.goal)
        #self.locals['new_obs'][0, 12:] = self.goal
        #self.locals['new_obs'][0, 0:3] = self.locals['new_obs'][0, 0:3] - np.reshape(self.goal, self.locals['new_obs'][0, 0:3].shape)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print("rollout ended: callback entered")
        self.high_level_planner.reset()

    def set_goal(self, goal): 
        self.goal = goal

class RandGoalsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    This class is designed to provide a new random goal 
    during callback for training

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, waypt_gap_m=10.0, goal_tol_m=1.0, verbose=0):
    # def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(RandGoalsCallback, self).__init__(verbose)
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
        print("Setting rand goals for training...")
        self.waypt_gap_m=waypt_gap_m # max distance between the gen goal n the robot at time of gen
        self.goal_tol_m=goal_tol_m # min dist to consider the goal is met
        self.goal=np.array([5.0, 5.0, 5.0], dtype=np.float32)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        new_goal=copy.deepcopy(self.goal)
        self.training_env.set_goal(new_goal)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # print("global keys", self.globals.keys())
        # print("local keys", self.locals.keys())

        # check if near goal
        # TODO: check if this is the correct tensor?
        obs_tensor=(self.locals["new_obs"])
        print("obs: ", self.locals["new_obs"])


        # obs_tensor=(self.locals["obs_tensor"]).numpy()
        pos=obs_tensor[0,:3]
        # goal=obs_tensor[0,-3:]
        dist=np.sqrt(np.sum((self.goal-pos) ** 2))

        if dist < self.goal_tol_m: # if reached goal
            # set a new goal
            offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            # if it is too close to curr pose
            while np.any(offset < 1.0):
                # TODO: I could make this more efficient but I dun feel like it...
                offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            new_goal= pos+offset

            # check bounds
            if np.any(new_goal > 14.0) or np.any(new_goal < -14.0) :
                new_goal[0]= new_goal[0]-10.0
            new_goal[2]=np.random.uniform(0, 7.0)
            print("new goal ", new_goal)

            self.goal=np.array(new_goal, dtype=np.float32)
            # print("new obs: ", self.locals["obs_tensor"])

            # new_obs=copy.deepcopy(obs_tensor)
            # new_obs[0,-3:]=self.goal
            # self.locals["new_obs"]=new_obs
            # self.locals["obs_tensor"]=torch.from_numpy(new_obs)
            self.training_env.set_goal(self.goal)


        # self.locals["obs_tensor"]=torch.from_numpy(new_obs)
        # self.locals["new_obs"]=new_obs

        # print("new obs ", self.locals["obs_tensor"])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass