import sys, random
from enum import Enum
import torch
import numpy as np
import math
import os

class Case_Content(Enum):
    EMPTY = 1
    BODY = 2
    HEAD = 3
    APPLE = 4

class Action(Enum):
    DOWN = [1, 0]
    UP = [-1, 0]
    RIGHT = [0, 1]
    LEFT = [0, -1]
    
class Rewards(Enum):
    DEATH = -100
    ALIVE = 0
    CLOSER = 1
    FURTHER = -1
    APPLE = 100

class Env:

    def __init__(self, size):

        self.eaten = False
        self.game_over = False

        self.iteration = 0
        self.episode = 0
        self.max_score = 0
        self.score = 0
        self.size = size

        #grid = [[1]*(self.size+1)]*(self.size+1)
        #self.grid = torch.Tensor(grid)
        self.grid = torch.ones([21,21])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.snake = [[self.size // 2, self.size // 2]]
        self.grid[self.size // 2][self.size // 2] = Case_Content.HEAD.value
        self.apple = self.create_apple()

    def empty_blocks(self):
        emptys = []
        for i in range(1, self.size):
            for j in range(1, self.size):
                if(self.grid[i][j] == Case_Content.EMPTY.value):
                    emptys.append([i,j])
        return emptys   

    def create_apple(self):
        blocks = self.empty_blocks()
        apple = random.choice(blocks)
        self.grid[apple[0], apple[1]] = Case_Content.APPLE.value
        return apple

    def move_snake(self, action):
        if (len(self.snake) > 1):
            for block in self.snake:
                self.grid[block[0], block[1]] = Case_Content.BODY.value

        self.grid[self.snake[-1][0]+action.value[0], self.snake[-1][1]+action.value[1]] = Case_Content.HEAD.value
        self.snake.append([self.snake[-1][0]+action.value[0], self.snake[-1][1]+action.value[1]])
        
        if (self.snake[-1] != self.apple):
            if self.snake[-1] in self.snake[0:-2] or self.snake[-1][0] in [-1,self.size] or self.snake[-1][1] in [-1,self.size]:
                self.die()
                if (self.score > self.max_score):
                    self.max_score = self.score
            lost_tail = self.snake.pop(0)
            self.grid[lost_tail[0], lost_tail[1]] = Case_Content.EMPTY.value
        else:
            self.eaten = True
            self.score += 1
            self.grid[self.apple[0], self.apple[1]] = Case_Content.HEAD.value
            self.apple = self.create_apple()

    def die(self):
        self.game_over = True

    def calcul_distance(self, head, apple):
        return math.sqrt(pow(head[0] - apple[0], 2) + pow(head[1] - apple[1], 2))

    def step(self, action, state):
        prestate = state.clone()
        head_before = self.snake[-1].copy()
        self.iteration += 1
        
        reward = torch.tensor(Rewards.ALIVE.value)
        self.move_snake(action)
        if self.iteration > 100:
            self.game_over = True
        if(self.game_over):
            reward = torch.tensor(Rewards.DEATH.value)
            return prestate, prestate, reward, self.game_over
        
        elif(self.eaten):
            self.iteration = 0
            reward = torch.tensor(Rewards.APPLE.value)
            self.eaten = False
        else:
            if len(self.snake) < 5:
                distance_before = self.calcul_distance(head_before, self.apple)
                head_after = self.snake[-1]
                distance_after = self.calcul_distance(head_after, self.apple)
                if distance_after < distance_before:
                    reward = torch.tensor(Rewards.CLOSER.value)
                else:
                    reward = torch.tensor(Rewards.FURTHER.value)
                
        
        return prestate, self.get_state(), reward.to(self.device), self.game_over

    def get_state(self):

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        state = []
        for x, y in directions:
            wall_distance = 0
            apple_distance = 0
            body_distance = 0
            new_x = self.snake[-1][0] + x
            new_y = self.snake[-1][1] + y
            distance = 1
            while wall_distance == 0:
                if new_x in [-1, self.size+1] or new_y in [-1, self.size+1]:
                    wall_distance = distance
                else:
                    if [new_x, new_y] == self.apple:
                        apple_distance = distance
                    elif self.grid[new_x][new_y] == Case_Content.BODY:
                        body_distance = distance
                    distance += 1
                    new_x += x
                    new_y += y

            state.append([wall_distance, apple_distance, body_distance])
            #state.append(wall_distance)
            #state.append(apple_distance)
            #state.append(body_distance)
        
        return torch.tensor(state, dtype=torch.long).to(self.device)

    def reset(self):

        self.grid = torch.ones(self.grid.size())
        # Init game attributes
        self.game_over = False
        self.score = 0
        self.iteration = 0
        self.episode += 1
        # Create Snake
        self.snake = [[self.size // 2, self.size // 2]]
        # Update grid with snake position
        self.grid[self.size // 2, self.size // 2] = Case_Content.HEAD.value
        # Create Apple
        self.apple = self.create_apple()