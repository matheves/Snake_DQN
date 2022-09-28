# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:44:11 2021

@author: lele8
"""

import sys, pygame, random

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PANNEL_HEIGHT = 100
WINDOW_WIDTH = 400
WINDOW_HEIGHT = WINDOW_WIDTH + PANNEL_HEIGHT


BLOCKSIZE = 20 #Set the size of the grid block


def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.update()
    pygame.display.set_caption('Snake IA')
    game_over=False
    
    SCREEN.fill(WHITE)
    #drawGrid()
    rect = pygame.Rect(0, PANNEL_HEIGHT, WINDOW_WIDTH, WINDOW_WIDTH)
    pygame.draw.rect(SCREEN, BLACK, rect)
    
   
    score = 0
    drawScore(score)
    
    x1_change = 0       
    y1_change = 0
    
    snake = [[10,10]]
    apple = createApple(snake)
    
    CLOCK = pygame.time.Clock()
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -1
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = 1
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -1
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = 1
                    x1_change = 0
                          
        
        snake.append([snake[-1][0]+x1_change, snake[-1][1]+y1_change])
        drawRect(snake[-1][0], snake[-1][1], GREEN)
        
        if(snake[-1] != apple):
            if snake[-1] in snake[0:-2] or snake[-1][0] in [-1,WINDOW_WIDTH//BLOCKSIZE] or snake[-1][1] in [-1,WINDOW_WIDTH//BLOCKSIZE]:
                SCREEN.fill(BLACK)
                game_over = True
                print("Score:" + str(score))
            lost_tail = snake.pop(0)
            drawRect(lost_tail[0], lost_tail[1], BLACK)
        else:
            score += 1
            drawScore(score)
            apple = createApple(snake)
        
        pygame.display.update()
            
        CLOCK.tick(8)
     
    pygame.quit()
  
def createApple(snake):
    apple = [-1, -1]
    applex = -1
    appley = -1
    while apple == [-1, -1] or apple in snake:
        applex = random.randint(0, WINDOW_WIDTH//BLOCKSIZE-1)
        appley = random.randint(0, (WINDOW_HEIGHT-PANNEL_HEIGHT)//BLOCKSIZE-1)
        apple = [applex, appley]
    print(apple)
    drawRect(apple[0], apple[1], RED)
    return apple


def drawGrid():
    for x in range(WINDOW_WIDTH):
        for y in range(WINDOW_HEIGHT):
            rect = pygame.Rect(x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)
            
def drawRect(x, y, color):
    rect = pygame.Rect(x*BLOCKSIZE+1, y*BLOCKSIZE+1 + PANNEL_HEIGHT, BLOCKSIZE-2, BLOCKSIZE-2)
    pygame.draw.rect(SCREEN, color, rect)
 
            
def drawScore(score):
    rect = pygame.Rect(0, 0, WINDOW_WIDTH, PANNEL_HEIGHT)
    pygame.draw.rect(SCREEN, WHITE, rect)
    
    font = pygame.font.SysFont(None, 24)
    fontScore = font.render("Score: " + str(score), True, BLACK)
    SCREEN.blit(fontScore, (20, 20))
            
main()