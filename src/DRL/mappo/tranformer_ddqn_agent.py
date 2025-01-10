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
from configs.systemcfg import DEVICE

device = torch.device('cuda:'+str(DEVICE) if torch.cuda.is_available() else 'cpu')
if device == "cpu":
    print("cannot train with cpu")
    exit(0)
else:
    print("cuda: ", device)

# Double DQN model
    
from configs.systemcfg import ddqn_cfg
class DDQNAgent(nn.Module):
    global_memory = deque(maxlen=ddqn_cfg['maxlen_mem'])
    
    def __init__(self, state_size, action_size, checkpoint_path = './', load_model = False):
        super(DDQNAgent, self).__init__()
        self.load_model = load_model

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = ddqn_cfg['discount_factor']
        self.learning_rate = ddqn_cfg['learning_rate']
        self.epsilon = ddqn_cfg['epsilon']
        self.epsilon_decay = ddqn_cfg['epsilon_decay']
        self.epsilon_min =ddqn_cfg['epsilon_min']
        self.batch_size = ddqn_cfg['batch_size']
        self.train_start = self.batch_size
        # self.memory = DDQNAgent.global_memory
        self.memory = deque(maxlen=ddqn_cfg['maxlen_mem'])

        self.model = self.build_model().to(device)
        self.target_model = self.build_model().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        self.model_file = checkpoint_path 

        if self.load_model:
            self.model.load_state_dict(torch.load(self.model_file))
            self.epsilon = 0

        self.update_target_model()
        self.lock = Lock()

    def build_model(self):
        layer_size_1 = self.state_size + int(self.state_size*0.3)
        layer_size_2 = int(self.state_size*0.6)        
        layer_size_3 = int(self.state_size*0.2)


        model = nn.Sequential(
            nn.Linear(self.state_size, layer_size_1),
            nn.SELU(),
            nn.Linear(layer_size_1, layer_size_2),
            nn.SELU(),
            nn.Linear(layer_size_2, layer_size_3),
            nn.ELU(),
            nn.Linear(layer_size_3, self.action_size),
            nn.ELU()
        )
        self.softmax = nn.Softmax(dim=1)
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
            return np.random.randint(0, self.action_size)
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

    def train_model(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("self.learning_rate {}".format(self.learning_rate))
        # print("epsilon = {}".format(self.epsilon))
        self.lock.acquire()
        try:
            mini_batch = random.sample(self.memory, self.batch_size)

            states = np.zeros((self.batch_size, self.state_size))
            next_states = np.zeros((self.batch_size, self.state_size))
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

            target = self.model(states).to(device)
            target_val = self.target_model(next_states).to(device)

            for i in range(self.batch_size):
                target[i][actions[i]] = rewards[i] + self.discount_factor * torch.max(target_val[i])
            self.optimizer.zero_grad()
            loss = self.criterion(target, self.model(states)).to(device)
            loss.backward()
            self.optimizer.step()
            time.sleep(random.uniform(0.1, 1.0))
        finally:
            self.lock.release()
        # self.scheduler.step(loss.item())

        # self.learning_rate = self.optimizer.param_groups[0]['lr']
        # self.lock=0
        # return loss.item()

    def quantile_huber_loss(self, y_true, y_pred):
        quantiles = torch.linspace(1 / (2 * self.action_size), 1 - 1 / (2 * self.action_size), self.action_size).to(device)
        batch_size = y_pred.size(0)
        tau = quantiles.repeat(batch_size, 1)
        e = y_true - y_pred
        huber_loss = torch.where(torch.abs(e) < 0.5, 0.5 * e ** 2, torch.abs(e) - 0.5)
        quantile_loss = torch.abs(tau - (e < 0).float()) * huber_loss
        return quantile_loss.mean()
