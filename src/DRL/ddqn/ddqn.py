import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
# torch.autograd.set_detect_anomaly(True)
# Check if GPU is available
import time
from threading import Lock, active_count

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop, AdamW

import numpy as np

from DRL.common.agent import Agent
from DRL.common.model import ActorNetwork
from DRL.common.utils import identity, to_tensor_var

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')) 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
sys.path.append(".")

from configs.systemcfg import ddqn_cfg

device = None
class DDQN(Agent):
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, infor_name = "A2C_train_infor.txt", tdevice = "cuda:0"):
        super(DDQN, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda, infor_name=infor_name)
        device = tdevice

        self.discount_factor = ddqn_cfg['discount_factor']
        self.learning_rate = ddqn_cfg['learning_rate']
        self.epsilon = ddqn_cfg['epsilon']
        self.epsilon_min =ddqn_cfg['epsilon_min']
        self._ntrains = 0

        self.model = self.build_model().to(tdevice)
        self.target_model = self.build_model().to(tdevice)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.reduce_ep_start = False
        self.writte_ep = False
        self.update_target_model()
        

    def build_model(self):
        layer_size_1 = self.state_dim + int(self.state_dim*0.3)
        layer_size_2 = int(self.state_dim*0.6)        
        layer_size_3 = int(self.state_dim*0.2)
        model = nn.Sequential(
            nn.Linear(self.state_dim, layer_size_1),
            nn.SELU(),
            nn.Linear(layer_size_1, layer_size_2),
            nn.SELU(),
            nn.Linear(layer_size_2, layer_size_3),
            nn.ELU(),
            nn.Linear(layer_size_3, self.action_dim),
            nn.ELU()
        )
        # self.softmax = nn.Softmax(dim=1)
        return model
    
    def forward(self, x):
        x = self.model(x)
        #x = self.softmax(x)
        return x

    def save_model(self, name):
        torch.save(self.model.state_dict(), name)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_action(self, state, idx):
        if self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_dim)
        else:
            state = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_value = self.model(state)
            return torch.argmax(q_value).item()
        
    def get_actions(self, state, vid):
         # Generates actions and log probs from current Normal distribution.
        state = state.to(device)
        with torch.no_grad():
            actions = \
                self(state)
        actions = actions.cpu().detach()
        actions  = [vid, actions]
        return actions, None

    def add_memory(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        mini_batch = self.memory.sample_ddqn(self.batch_size)
        if not mini_batch:
            return
        self.reduce_ep_start = True
        self._ntrains += 1
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     self.log("self.learning_rate {}".format(self.learning_rate))
        # print("epsilon = {}".format(self.epsilon))
        
        states = np.zeros((self.batch_size, self.state_dim))
        next_states = np.zeros((self.batch_size, self.state_dim))
        actions, rewards = [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        predicted_q = self.model(states)
        target_val = self.target_model(next_states).detach()

        max_next_q = target_val.max(dim=1)[0]
        target = predicted_q.clone()  

        for i in range(self.batch_size):
            for j in range(len(actions[i])):  
                target[i][j] = rewards[i] + self.discount_factor * max_next_q[i]
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_q, target)
        loss.backward()
        self.optimizer.step()
       
    def quantile_huber_loss(self, y_true, y_pred):
        quantiles = torch.linspace(1 / (2 * self.action_dim), 1 - 1 / (2 * self.action_dim), self.action_dim).to(device)
        batch_size = y_pred.size(0)
        tau = quantiles.repeat(batch_size, 1)
        e = y_true - y_pred
        huber_loss = torch.where(torch.abs(e) < 0.5, 0.5 * e ** 2, torch.abs(e) - 0.5)
        quantile_loss = torch.abs(tau - (e < 0).float()) * huber_loss
        return quantile_loss.mean()
    def exploration_action(self, state):
        epsilon = self.epsilon_start
        if self.reduce_ep_start:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                    np.exp(-1. * (self._ntrains) / self.epsilon_decay)
        if epsilon <= self.epsilon_end and not self.writte_ep:
            self.writte_ep = True
            self.log("Epsilon is not change at epoche, step {} {}".format(self.n_episodes, self.n_steps))
        elif not self.writte_ep:
            self.log("epsilon {} at epoch, steps {} {}, self._ntrains {}".format(epsilon,self.n_episodes, self.n_steps, self._ntrains))                     
        if np.random.rand() < epsilon:
            action =[]
            for i in range(len(self.env.vehicles)):
                num_m = len(self.env.missions)//len(self.env.vehicles) #num_mission per vehicle
                label = [i]*num_m
                order = list(range(num_m))
                np.random.shuffle(order)
                action += list(zip(order,label))
            np.random.shuffle(action)
        else:
            action = self.action(state)
        return action
    
    # Choose actions based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.model(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else: 
            state_action_value = state_action_value_var.data.numpy()[0]
        state_action_value = state_action_value.flatten()
        action = self.convert_actions_new(state_action_value, len(self.env.vehicles))
        # action = np.argmax(state_action_value)
        return action
    
    def convert_actions_new(self, sm_actions, num):
        if not isinstance(sm_actions, (list, np.ndarray)):
            sm_actions = [sm_actions]

        if len(sm_actions) == 0:
            return []
        self.reduce_ep_start = True
        indexed_data = [(value, index) for index, value in enumerate(sm_actions)]
        sorted_data = sorted(indexed_data, key=lambda x: x[0])
        num_task = int(len(sm_actions)/num)
        labeled_data = []
        idx_cnt = [0]*num
        for i in range(0, len(sorted_data), num_task):
            chunk = sorted_data[i:i+10]
            label = i // 10
            for value, original_index in chunk:
                labeled_data.append((original_index, value, label, idx_cnt[label]))
                idx_cnt[label] += 1
        labeled_data.sort(key=lambda x: x[0])

        labels = []
        for original_index, value, label, order in labeled_data:
            labels.append((order,label))
        return labels
    def log(self, message):
        pass
        # with open(self.infor, "a") as file:
        #     file.write(message + "\n")

    # agent interact with the environment to collect experience
    def interact(self, current_datetime = None, rfile = False):
        super(DDQN, self)._take_one_step(current_datetime = current_datetime, rfile = rfile)


