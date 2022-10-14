# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:47:40 2022

@author: lele8
"""

import torch
import random
from collections import  deque



class DQN(torch.nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.outputs = outputs
        
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm3d(16)
        self.conv2 = torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm3d(32)
        #self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        #self.bn3 = torch.nn.BatchNorm2d(32)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = "cuda:0"
        self.device = "cpu"
        
    
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        #linear_input_size = convw * convh * 32
        #self.head = torch.nn.Linear(linear_input_size * 4, outputs)

    def forward(self, x):
        x = x.to(self.device)
        #x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.conv1(x))
        #x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        self.head = torch.nn.Linear(x.shape[1], self.outputs)
        #x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x

class DQN_Snake:

    def __init__(self, height, width, n_outputs, env):
        self.n_outputs = n_outputs
        self.env = env
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.episode_duration = []
        self.model = DQN(height, width, self.n_outputs).to(self.device)

        self.batch_size = 6
        self.gamma = 0.95
        self.learning_rate = 1e-2

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)
        self.replay_memory = deque(maxlen=2000)
        self.step_done = 0
        

        self.loss_fn = torch.nn.MSELoss()
        
        

    def epsilon_greedy_policy(self, state, epsilon=0):
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.n_outputs, (1,1))
        else:
            Q_values = torch.tensor([[torch.argmax(self.model(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(state, 0), 0), 0)))]])
            return Q_values
        
    def sample_experiences(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        states = torch.unsqueeze(batch[0][0].detach().clone(), 0)
        
        actions = torch.unsqueeze(batch[0][1].detach().clone(), 0)
        rewards = torch.unsqueeze(torch.tensor([batch[0][2]]), 0)
        next_states = torch.unsqueeze(batch[0][3].detach().clone(), 0)
        dones = torch.unsqueeze(torch.tensor([batch[0][4]]), 0)
        
        
        for i in range(1, len(batch)):
            states = torch.cat((states, torch.unsqueeze(batch[i][0].detach().clone(), 0)))
            actions = torch.cat((actions, torch.unsqueeze(batch[i][1].detach().clone(), 0)))
            rewards = torch.cat((rewards, torch.unsqueeze(torch.tensor([batch[i][2]]), 0)))
            next_states = torch.cat((next_states, torch.unsqueeze(batch[i][3].detach().clone(), 0)))
            dones = torch.cat((dones, torch.unsqueeze(torch.tensor([batch[i][4]]), 0)))
        
        
        #rewards = batch[2]
        #next_states = batch[3]
        #dones = batch[4]
        
        
        #states, actions, rewards, next_states, dones = [
        #    torch.tensor([experience[field_index] for experience in batch])
        #    for field_index in range(5)]
        return states, actions, rewards, next_states, dones

        
    def play_one_step(self, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        prestate, next_state, reward, done, score = self.env.step(action)
        self.replay_memory.append([prestate, action, reward, next_state, done])
        return next_state, reward, done, score
        
    
    def training_step(self):
        states, actions, rewards, next_states, dones = self.sample_experiences()
        
        next_Q_values = self.model(torch.unsqueeze(torch.unsqueeze(next_states, 0),0))

        max_next_Q_values = torch.max(next_Q_values, axis=1)
 
        target_Q_values = (rewards +
                           (1 - dones[max_next_Q_values.indices]) * self.gamma * max_next_Q_values.values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = torch.nn.functional.one_hot(actions.to(torch.int64), self.n_outputs)
        predict = self.model(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(states[0], 0), 0),0))
        Q_values = torch.sum(predict * mask[0], axis=1, keepdim=True)
        loss = self.loss_fn(target_Q_values, Q_values)
        
        for i in range(1, len(states)):
            predict = self.model(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(states[i], 0), 0),0))
            Q_values = torch.sum(predict * mask[i], axis=1, keepdim=True)
            loss += self.loss_fn(target_Q_values, Q_values)
        loss = torch.mean(loss)

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
           
    
    
