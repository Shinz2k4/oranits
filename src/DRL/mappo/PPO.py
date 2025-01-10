import torch as th
from torch import nn
from torch.optim import Adam, RMSprop, AdamW

import numpy as np
from copy import deepcopy

from DRL.common.agent import Agent
from DRL.common.model import ActorNetwork, CriticNetwork
from DRL.common.utils import index_to_one_hot, to_tensor_var
import pandas as pd

import matplotlib.pyplot as plt
class PPO(Agent):
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True,
                 infor_name = "A2C_train_infor.txt"):
        super(PPO, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda, infor_name=infor_name)

        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "adamw":
            self.actor_optimizer = AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-4)
            self.critic_optimizer = AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
        
        self.writte_ep = False
        self.reduce_ep_start = False
        
        self.df_loss = {"actor":[], "critic":[]}

    def save_model(self, filepath):
        th.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer_type.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = th.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_type.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")

    def log(self, message):
        with open(self.infor, "a") as file:
            file.write(message + "\n")
            
    # agent interact with the environment to collect experience
    def interact(self):
        super(PPO, self)._take_one_step()

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)

        states = np.stack(batch.states)
        states_var = to_tensor_var(states, self.use_cuda) # Shape: [10, 776]
        states_var = states_var.unsqueeze(1).expand(-1, 30, -1).contiguous()  # Shape: [10, 30, 776]
        states_var = states_var.view(-1, self.state_dim)  # Shape: [300, 776]
        
        actions = np.stack(batch.actions)
        actions = actions.astype(int)
        one_hot_actions = index_to_one_hot(actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)

        # rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        rewards = np.array(batch.rewards).astype(float)
        rewards_var = to_tensor_var(rewards, self.use_cuda)  # Shape: [10, 1]
        rewards_var = rewards_var.repeat(1, 30).view(-1, 1)   # Shape: [300, 1]

        # update actor network
        self.actor_optimizer.zero_grad()
        values = self.critic_target(states_var, actions_var).detach()
        advantages = rewards_var - values
        # # normalizing advantages seems not working correctly here
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        action_log_probs = self.actor(states_var)
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        old_action_log_probs = self.actor_target(states_var).detach()
        old_action_log_probs = th.sum(old_action_log_probs * actions_var, 1)
        ratio = th.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        # PPO's pessimistic surrogate (L^CLIP)
        actor_loss = -th.mean(th.min(surr1, surr2))
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        values = self.critic(states_var, actions_var)
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)/self.batch_size
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # update actor target network and critic target network
        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            super(PPO, self)._soft_update_target(self.actor_target, self.actor)
            super(PPO, self)._soft_update_target(self.critic_target, self.critic)

    def plot_save_loss(self, path):
        plt.figure(figsize=(8, 5))  
        plt.plot(self.df_loss["actor"], label="Actor Loss", marker='o', linestyle='-')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path+"/loss_actor.png", dpi = 300)
        plt.close()
        
        
        plt.figure(figsize=(8, 5)) 
        plt.plot(self.df_loss["critic"], label="Critic Loss", marker='s', linestyle='--')
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path+"/loss_critic.png", dpi = 300)
        plt.close()
        
        df = pd.DataFrame(self.df_loss)
        df.to_csv(path+"/loss_actor_critic.csv", index=None)

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy() 
        else:
            softmax_action = softmax_action_var.data.numpy()
        return softmax_action
    
    def convert_actions_new(self, sm_actions, num):
        if not isinstance(sm_actions, (list, np.ndarray)):
            sm_actions = [sm_actions]

        if len(sm_actions) == 0:
            return []

        indexed_data = [(value, index) for index, value in enumerate(sm_actions)]

        sorted_data = sorted(indexed_data, key=lambda x: x[0])

        num_task = int(len(sm_actions)/num)

        labeled_data = []
        for i in range(0, len(sorted_data), num_task):
            chunk = sorted_data[i:i+10]
            label = i // 10  
            for value, original_index in chunk:
                labeled_data.append((original_index, value, label))

        labeled_data.sort(key=lambda x: x[0])

        labels = []
        for original_index, value, label in labeled_data:
            labels.append(label)
        return labels
    
    # choose list of actions based on state with random noise added for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        epsilon = self.epsilon_start
        if self.reduce_ep_start:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                    np.exp(-1. * (self.n_steps-self.reduce_ep_start) / self.epsilon_decay)
        if epsilon <= self.epsilon_end and not self.writte_ep:
            self.writte_ep = True
            self.log("Epsilon is not change at epoche, step {} {}".format(self.n_episodes, self.n_steps))
        elif not self.writte_ep:
            self.log("epsilon {} at epoch, steps {} {}".format(epsilon,self.n_episodes, self.n_steps))                     
            
        if np.any(np.isnan(softmax_action)):
            print( softmax_action)
            softmax_action = self._softmax_action(state)
        if np.random.rand() < epsilon:
            action = np.random.randint(0, len(self.env.vehicles), size = len(self.env.missions))
        else:
            softmax_action = softmax_action.flatten()
            action = self.convert_actions_new(softmax_action, len(self.env.vehicles))
        return action
    
    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        softmax_action = softmax_action.flatten()
        action = self.convert_actions_new(softmax_action, len(self.env.vehicles))
        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value