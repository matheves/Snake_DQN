from binascii import a2b_base64
from numpy import dtype
import torch
import random
import math
import numpy as np
from collections import namedtuple, deque
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, x):
        """Save a transition"""
        self.memory.append(x)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(DQN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.flatten = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(D_in, H)
        self.lin2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
          x : [batch_size, 1, height, width ]
        """
        x = x.flatten().to(self.device).float()
        x = self.lin1(x)
        x = torch.nn.functional.relu(x) 
        x = self.lin2(x)
    
        return x

class DQN_Snake:

    BATCH_SIZE = 12
    GAMMA = 0.95
    EPS_START = 0.995
    EPS_END = 0.01
    EPS_DECAY = 20000
    LEARNING_RATE = 0.00025

    def __init__(self, D_in, H, D_out):
        self.n_actions = D_out
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.episode_duration = []
        self.dqn = DQN(D_in, H, D_out).to(self.device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr= self.LEARNING_RATE)
        self.memory = ReplayMemory(10000)
        self.step_done = 0
        self.loss_fn = torch.nn.MSELoss()

    def save_model(self):
        torch.save(self.dqn.state_dict(), "./sam_model.pt")

    def save_optimizer(self):
        torch.save(self.optimizer.state_dict(), "./sam_optimizer.pt")

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))

    def load_optimizer(self, path):
        self.optimizer.load_state_dict(torch.load(path))

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step_done / self.EPS_DECAY)
        self.step_done += 1
        if sample > eps_threshold:
            #return torch.argmax(self.dqn(state))
            
            return torch.tensor([[torch.argmax(self.dqn(state))]], device=self.device, dtype=torch.int32)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.int32)
            #return torch.tensor(random.randrange(self.n_actions))

    def train_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            #print("Missing some state to start training")
            return
        #print("Start training")
        transitions = self.memory.sample(self.BATCH_SIZE)
        
        states = torch.tensor([])
        actions = torch.tensor([])
        rewards = torch.tensor([])
        next_Q_values = torch.tensor([])
        dones = torch.tensor([])
        
        for t in transitions:
            state = t[0]
            action = t[1]
            reward = t[2]
            next_state = t[3]
            done = t[4]
            
            #ajouter de la concat ou du stack de tensor
            states = torch.cat((states,torch.unsqueeze(state,0)))
            actions = torch.cat((actions,torch.unsqueeze(action,0)))
            rewards = torch.cat((rewards,torch.unsqueeze(torch.tensor([reward]),0)))
            next_Q_values = torch.cat((next_Q_values,torch.unsqueeze(self.dqn(next_state),0)))
            dones = torch.cat((dones,torch.unsqueeze(torch.tensor([done]),0)))
        
        max_next_Q_values = torch.max(next_Q_values, axis=1)
        
        target_Q_values = (rewards +
                           (1 - dones[max_next_Q_values.indices]) * self.GAMMA * max_next_Q_values.values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = torch.nn.functional.one_hot(actions.to(torch.int64), self.n_actions)
        predict = self.dqn(states[0])
        Q_values = torch.sum(predict * mask[0], axis=1, keepdim=True)
        loss = self.loss_fn(target_Q_values, Q_values)
        
        for i in range(1, len(states)):
            predict = self.dqn(states[i])
            Q_values = torch.sum(predict * mask[i], axis=1, keepdim=True)
            loss += self.loss_fn(target_Q_values, Q_values)
        loss = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
