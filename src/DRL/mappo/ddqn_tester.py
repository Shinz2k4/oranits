import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import os
import threading
from threading import active_count
import sys
from physic_definition.system_base.ITS_based import TaskGenerator
# plt.style.use('dark_background')
import copy

class DDQNTester:

    def __init__(self, env, agents):
        # Initialize relevant variables for training.
        self.env = env
        self.agents = agents
        self.tg = TaskGenerator(15,env.lmap)
    def eval(self):
        """
        Runs a single episode in the training process for max_episode_length
        timesteps.

        Returns:
            scores: List of rewards gained at each timestep.
        """

        # Initialize list to hold reward values at each timestep.
        scores = []
        for i in range(3):
            scores.append([])

     
        env_info = self.env.reset()
        states = env_info[0]

        for idx, state in enumerate(states):
            agent = self.agents[idx]
            observationip = np.reshape(states[state], (1, -1))
            processed_state = torch.from_numpy(observationip).float()
            action, log_prob = agent.get_actions(processed_state, idx)
           
  

    