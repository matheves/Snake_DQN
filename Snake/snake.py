# -*- coding: utf-8 -*-

import sys, pygame, random
from enum import Enum
import torch


class Case_Content(Enum):
    EMPTY = 1
    BODY = 2
    HEAD = 3
    APPLE = 4

class Action(Enum):
    DOWN = 1
    UP = 2
    RIGHT = 3
    LEFT = 4

class Snake_Game():

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    def __init__(self, pannel_height, width, block_size):
        # init layout and grid parameters
        self.height_pannel = pannel_height
        self.height_screen = width + self.height_pannel
        self.width_screen = width
        self.block_size = block_size
        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self.width_screen, self.height_screen))
        pygame.display.set_caption("DQL Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()
        pygame.display.update()
        

    def create_apple(self):
        apple = [-1, -1]
        applex = -1
        appley = -1
        while apple == [-1, -1] or apple in self.snake:
            applex = random.randint(0, self.width_grid-1)
            appley = random.randint(0, self.height_grid-1)
            apple = [applex, appley]
        self.grid[apple[1], apple[0]] = Case_Content.APPLE.value
        self.drawRect(apple[0], apple[1], self.RED)
        return apple

    def drawGrid(self):
        for x in range(self.width_screen):
            for y in range(self.height_screen):
                rect = pygame.Rect(x*self.block_size, y*self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)
            
    def drawRect(self, x, y, color):
        rect = pygame.Rect(x*self.block_size+1, y*self.block_size+1 + self.height_pannel, self.block_size-2, self.block_size-2)
        pygame.draw.rect(self.screen, color, rect)
 
            
    def drawScore(self):
        rect = pygame.Rect(0, 0, self.width_screen, self.height_pannel)
        pygame.draw.rect(self.screen, self.WHITE, rect)
    
        font = pygame.font.SysFont(None, 24)
        fontScore = font.render("Score: " + str(self.score), True, self.BLACK)
        self.screen.blit(fontScore, (20, 20))

    def move_snake(self, x, y):
        if (len(self.snake) > 1):
            for block in self.snake:
                self.grid[block[1], block[0]] = Case_Content.BODY.value

        self.grid[self.snake[-1][1]+y, self.snake[-1][0]+x] = Case_Content.HEAD.value
        self.snake.append([self.snake[-1][0]+x, self.snake[-1][1]+y])
        self.drawRect(self.snake[-1][0], self.snake[-1][1], self.GREEN)

        if (self.snake[-1] != self.apple):
            if self.snake[-1] in self.snake[0:-2] or self.snake[-1][0] in [-1,self.width_grid] or self.snake[-1][1] in [-1,self.width_grid]:
                self.screen.fill(self.BLACK)
                self.game_over = True
                print("Score:" + str(self.score))
            lost_tail = self.snake.pop(0)
            self.grid[lost_tail[1], lost_tail[0]] = Case_Content.EMPTY.value
            self.drawRect(lost_tail[0], lost_tail[1],self.BLACK)
        else:
            self.score += 1
            self.drawScore()
            self.grid[self.apple[1], self.apple[0]] = Case_Content.HEAD.value
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
        self.grid = torch.IntTensor(grid)

        # Init game attributes
        self.game_over = False
        self.score = 0
        # Create Snake
        self.snake = [[self.height_grid // 2, self.width_grid // 2]]
        # Update grid with snake position
        self.grid[self.height_grid // 2, self.width_grid // 2] = Case_Content.HEAD.value
        # Create Apple
        self.apple = self.create_apple()
        # Update grid with apple position
        self.grid[self.apple[1], self.apple[0]] = Case_Content.APPLE.value
        self.drawScore()

   