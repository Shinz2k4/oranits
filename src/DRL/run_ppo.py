import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from datetime import datetime

from DRL.mappo.PPO import PPO
from DRL.common.utils import agg_double_list
from DRL.rl_env import *
import sys
import gymnasium as gym
import numpy as np
import pandas as pd

import shutil

current_file = __file__


from configs.systemcfg import log_configs, verbose, mission_cfg, map_cfg, ppo_cfg, DEVICE
from utils import Load, write_config
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)


MAX_EPISODES = 15000
EPISODES_BEFORE_TRAIN = 500
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# roll out n steps
ROLL_OUT_N_STEPS = 30
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = 10000000
# only use the latest ROLL_OUT_N_STEPS for training PPO
BATCH_SIZE = 128

TARGET_UPDATE_STEPS = 5  #finetune
TARGET_TAU = 1.0

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None
CRITIC_LOSS = "huber"
OPTIMIZER_TYPE = "adamw"

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 100

RANDOM_SEED = 2024

ACTOR_LR=0.001
CRITIC_LR=0.0001

current_datetime = datetime.now().strftime("%Y%m%d")
current_datetime_hour = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./output_{current_datetime}"
checkpoint_dir = f"{output_dir}/checkpoints/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

shutil.copy(current_file, output_dir)

def run():
    # Prepare data
    data = write_config()
    if data:
        config, graph, map = data
    else:
        print("Cannot read file!!!")
    register_env("its_env", lambda config: SITSEnv(config, verbose=verbose, map__=map))
    # Initialize environment, extract state/action dimensions and num agents.
    env = SITSEnv(config, verbose=verbose, map__=map, max_steps= 20)
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.shape[0]

    ppo = PPO(env=env, memory_capacity=MEMORY_CAPACITY,
              state_dim=state_size, action_dim=action_size,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
              target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
              episodes_before_train=EPISODES_BEFORE_TRAIN,
              critic_loss=CRITIC_LOSS,
              actor_lr=ACTOR_LR,
              critic_lr=CRITIC_LR,
              optimizer_type=OPTIMIZER_TYPE,
              infor_name = output_dir + "/ppo_infor.txt")
    
    episodes =[]
    eval_rewards =[]

    ppo.log("time to log {}".format(current_datetime_hour))
    while ppo.n_episodes < MAX_EPISODES:
        env.reset(current_datetime = current_datetime_hour)
        ppo.interact()
        if ppo.n_episodes >= EPISODES_BEFORE_TRAIN:
            ppo.train()
        if ppo.episode_done and ((ppo.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards, _ = ppo.evaluation(env, EVAL_EPISODES) #env_eval
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (ppo.n_episodes+1, rewards_mu))
            episodes.append(ppo.n_episodes+1)
            eval_rewards.append(rewards_mu)

            data_to_save = {
                'Episodes': episodes,
                'Average Reward': eval_rewards
            }
            df = pd.DataFrame(data_to_save)
            df.to_csv(f"{output_dir}/real_map_ppo_eval_data.csv", index=False)

            plt.figure()
            plt.plot(episodes, eval_rewards, label="ppo")
            plt.title("Real_map")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.legend()
            plt.savefig(f"{output_dir}/ppo_real_map_{ppo.n_episodes + 1}_episodes.png")
            plt.close()
            # ppo.save_model(f"{checkpoint_dir}/check_point_{ppo.n_episodes + 1}.pth")
            ppo.plot_save_loss(output_dir)
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()