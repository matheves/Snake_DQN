# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:18:57 2022

@author: lele8
"""

# -*- coding: utf-8 -*-

import sys, pygame, random
from enum import Enum
import torch
import numpy as np
import math

class Case_Content(Enum):
    EMPTY = 1
    BODY = 2
    HEAD = 3
    APPLE = 4
    
class Rewards(Enum):
    DEATH = -100
    ALIVE = -1
    CLOSER = 5
    APPLE = 100

class Snake_Game():

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    GREEN_HEAD = (0, 153, 76)

    def __init__(self, pannel_height, width, block_size, mode="human"):
        # init layout and grid parameters
        self.height_pannel = pannel_height
        self.height_screen = width + self.height_pannel
        self.width_screen = width
        self.block_size = block_size
        self.eaten = False
        self.mode = mode
        self.episode = -1
        self.iterations = 0
        self.actions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        # Pygame init      
        pygame.init()
        pygame.display.init()

        if self.mode == "human":
            self.screen = pygame.display.set_mode((self.width_screen, self.height_screen))
            pygame.display.set_caption("DQL Snake AI")
            pygame.display.update()

            
        self.clock = pygame.time.Clock()
        self.surf = pygame.Surface((self.width_screen, self.height_screen))
        self.surf.fill(self.BLACK)
        #rect = pygame.Rect(0, 100, self.width_screen, self.width_screen)
        #pygame.draw.rect(self.surf, self.BLACK, rect)
         
        self.reset()
        self.render()
        if self.mode == "human":
            pygame.display.update()
        
    def empty_blocks(self):
        emptys = []
        for i in range(self.height_grid):
            for j in range(self.width_grid):
                if(self.grid[i][j] == Case_Content.EMPTY.value):
                    emptys.append([i,j])
        return emptys   
    
    
    def create_apple(self):
        #self.drawRect(apple[0], apple[1], self.RED)
        #blocks = self.empty_blocks()
        #apple = random.choice(blocks)
        apple = self.snake[-1].copy()
        apple[0] += 1
        apple[1] += 1
        
        self.grid[apple[0], apple[1]] = Case_Content.APPLE.value
        self.drawRect(apple[1], apple[0], self.RED)
        return apple


    def drawGrid(self):
        for x in range(self.width_screen):
            for y in range(self.height_screen):
                rect = pygame.Rect(x*self.block_size, y*self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)
            
    def drawRect(self, x, y, color):
        rect = pygame.Rect(x*self.block_size+1, y*self.block_size+1 + self.height_pannel, self.block_size-2, self.block_size-2)
        pygame.draw.rect(self.surf, color, rect)
 
            
    def drawScore(self):
        rect = pygame.Rect(0, 0, self.width_screen, self.height_pannel)
        pygame.draw.rect(self.surf, self.WHITE, rect)
    
        font = pygame.font.SysFont(None, 24)
        fontScore = font.render("Episode: " + str(self.episode) + " - Score: " + str(self.score), True, self.BLACK)
        self.surf.blit(fontScore, (20, 20))

    def move_snake(self, action):
        
        self.grid[self.snake[-1][0]+self.actions[action][0], self.snake[-1][1]+self.actions[action][1]] = Case_Content.HEAD.value
        self.snake.append([self.snake[-1][0]+self.actions[action][0], self.snake[-1][1]+self.actions[action][1]])
        self.drawRect(self.snake[-1][1], self.snake[-1][0], self.GREEN_HEAD)
        if len(self.snake) > 1:    
            self.drawRect(self.snake[-2][1], self.snake[-2][0], self.GREEN)
        if (self.snake[-1] != self.apple):
            if self.snake[-1] in self.snake[0:-2] or self.snake[-1][0] in [-1,self.width_grid] or self.snake[-1][1] in [-1,self.width_grid]:
                self.surf.fill(self.BLACK)
                self.game_over = 1
                #print("Score:" + str(self.score))
            lost_tail = self.snake.pop(0)
            self.grid[lost_tail[0], lost_tail[1]] = Case_Content.EMPTY.value
            self.drawRect(lost_tail[1], lost_tail[0],self.BLACK)
        else:
            self.eaten = True
            self.score += 1
            #self.drawScore()
            self.grid[self.apple[0], self.apple[1]] = Case_Content.HEAD.value
            self.apple = self.create_apple()

    def update(self):
        pygame.display.update()

    def quit(self):
        pygame.quit()

    def get_events(self):
        return pygame.event.get()
    
    def reset(self):
        
        self.height_grid = (self.height_screen - self.height_pannel) // self.block_size
        self.width_grid = self.width_screen // self.block_size
        
        # Create grid as a tensor for our DQL model
        grid = [[1]*(self.width_grid+1)]*(self.height_grid+1)
        self.grid = torch.FloatTensor(grid)

        # Init game attributes
        self.iterations = 0
        self.game_over = 0
        self.score = 0
        self.episode += 1
        # Create Snake
        self.snake = [[self.height_grid // 2, self.width_grid // 2]]
        self.drawRect(self.snake[-1][1], self.snake[-1][0], self.GREEN_HEAD)
        # Update grid with snake position
        self.grid[self.height_grid // 2, self.width_grid // 2] = Case_Content.HEAD.value
        # Create Apple
        self.apple = self.create_apple()
            
        
        #self.drawScore()
        
        
    def calcul_distance(self, head, apple):
        return math.sqrt(pow(head[0] - apple[0], 2) + pow(head[1] - apple[1], 2))
        
    def step(self, action):
        prestate = self.render()
        
        head_before = self.snake[-1].copy()
        
        reward = Rewards.ALIVE.value
        self.move_snake(action)
        self.iterations += 1
        
        if self.iterations > 100:
            self.game_over = 1
        if self.game_over:
            reward = Rewards.DEATH.value
        
        elif self.eaten:
            self.iterations = 0
            reward = Rewards.APPLE.value
            self.eaten = False
        elif len(self.snake) < 5:
                distance_before = self.calcul_distance(head_before, self.apple)
                head_after = self.snake[-1]
                distance_after = self.calcul_distance(head_after, self.apple)
                if distance_after < distance_before:
                    reward = Rewards.CLOSER.value
                else:
                    reward = -Rewards.CLOSER.value
        return prestate, self.render(), reward, self.game_over, self.score

    def render(self):
        if self.mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            self.clock.tick(1000)
            pygame.display.update()
            
        elif self.mode == "rgb_array":
            #return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1,0,2))
            #img = pygame.surfarray.pixels3d(self.surf)
            red = pygame.surfarray.array_red(self.surf)
            green = pygame.surfarray.array_green(self.surf)
            blue = pygame.surfarray.array_blue(self.surf)
            
            return [red, green, blue]
        else:
            print("mode inconnu, need : [human, rgb_array]")
        