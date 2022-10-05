from DQN import DQN_Snake
from snake import Snake_Game
import pygame

PANNEL_HEIGHT = 100
WINDOW_WIDTH = 400
BLOCKSIZE = 20
TICK = 8
NUM_EPISODE = 50

game = Snake_Game(PANNEL_HEIGHT, WINDOW_WIDTH, BLOCKSIZE)

episode= 0
x_change = 0
y_change = 0

model = DQN_Snake(game.height_grid, game.width_grid, 5)

while episode < NUM_EPISODE:
    state = game.grid
    while not game.game_over:
        for event in game.get_events():
            if event.type == pygame.QUIT:
                game.game_over = True
                episode = NUM_EPISODE
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x_change = -1
                    y_change = 0
                elif event.key == pygame.K_RIGHT:
                    x_change = 1
                    y_change = 0
                elif event.key == pygame.K_UP:
                    x_change = 0
                    y_change = -1
                elif event.key == pygame.K_DOWN:
                    x_change = 0
                    y_change = 1
                        
        game.move_snake(x_change, y_change)
        game.update()
        game.clock.tick(TICK)
    episode += 1
    game.reset()
game.quit()

'''
        
    '''