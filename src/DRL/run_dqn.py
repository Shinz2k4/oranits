import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from datetime import datetime

from DRL.dqn.dqn import DQN
from DRL.common.utils import agg_double_list
from DRL.rl_env import *

import sys
import gymnasium as gym
import numpy as np
import pandas as pd

from configs.systemcfg import log_configs, verbose, mission_cfg, map_cfg, ppo_cfg, DEVICE
from utils import Load
import torch
import torch.optim as optim
import torch.nn as nn
import shutil

device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


MAX_EPISODES = 10000
EPISODES_BEFORE_TRAIN = 10
EVAL_EPISODES = 10
EVAL_INTERVAL = 1
current_file = __file__

# max steps in each episode, prevent from running too long
MAX_STEPS = None

MEMORY_CAPACITY = 10000000
BATCH_SIZE = 128
CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

REWARD_DISCOUNTED_GAMMA = 0.95

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

DONE_PENALTY = -10.

RANDOM_SEED = 2024

current_datetime = datetime.now().strftime("%Y%m%d")
current_datetime_hour = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./output_dqn_{current_datetime}_new_mechanism"
checkpoint_dir = f"{output_dir}/checkpoints/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
shutil.copy(current_file, output_dir)

def run():
    load = Load()
    mission_decoded_data, graph, map =  load.data_load()
    # Prepare data
    config = {
        "n_missions": mission_cfg['n_mission'],
        "n_vehicles": mission_cfg['n_vehicle'],
        "n_miss_per_vec": mission_cfg['n_miss_per_vec'],
        "decoded_data": mission_decoded_data,
        "segments": map.get_segments(),
        "graph": graph,
        "thread": 1,
        "detach_thread": 0,
        "score_window_size": 100,
        "tau": 10*6 #min
    }
    register_env("its_env", lambda config: SITSEnv(config, verbose=verbose, map__=map))
    # Initialize environment, extract state/action dimensions and num agents.
    env = SITSEnv(config, verbose=verbose, map__=map, max_steps= 20)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.shape[0]

    dqn = DQN(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_size, action_dim=action_size,
              batch_size=BATCH_SIZE, max_steps=MAX_STEPS,
              done_penalty=DONE_PENALTY, critic_loss=CRITIC_LOSS,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN,
              infor_name=output_dir+"/dqn_infor.txt"
              )

    episodes =[]
    eval_rewards =[]
    
    dqn.log("time to log {}".format(current_datetime_hour))
    while dqn.n_episodes < MAX_EPISODES:
        env.reset(current_datetime = current_datetime_hour, rfile = True)
        dqn.interact(current_datetime = current_datetime_hour, rfile = True)
        if dqn.n_episodes >= EPISODES_BEFORE_TRAIN:
            dqn.train()
        if dqn.episode_done and ((dqn.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards, _ = dqn.evaluation(env, EVAL_EPISODES) #env_eval
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (dqn.n_episodes+1, rewards_mu))
            episodes.append(dqn.n_episodes+1)
            eval_rewards.append(rewards_mu)

            data_to_save = {
                'Episodes': episodes,
                'Average Reward': eval_rewards
            }
            df = pd.DataFrame(data_to_save)
            df.to_csv(f"{output_dir}/real_map_dqn_eval_data.csv", index=False)

            plt.figure()
            plt.plot(episodes, eval_rewards, label="DQN")
            plt.title("Real_map")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.savefig(f"{output_dir}/dqn_real_map_{dqn.n_episodes + 1}_episodes.png")
            plt.close()  # Close the figure to save memory
    # dqn.save_model(f"{output_dir}/real_map_dqn_model.pth")
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()