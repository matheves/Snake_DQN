from DQN import DQN_Snake
from snake2 import Action
from snake2 import Snake_Game
import pygame

PANNEL_HEIGHT = 100
WINDOW_WIDTH = 400
BLOCKSIZE = 20
TICK = 8
NUM_EPISODE = 50

game = Snake_Game(PANNEL_HEIGHT, WINDOW_WIDTH, BLOCKSIZE, "human")

episode= 0
x_change = 0
y_change = 0

model = DQN_Snake(game.height_grid, game.width_grid, 5)


while episode < NUM_EPISODE:
    state = game.grid
    action = Action.STATIC
    while not game.game_over:
        for event in game.get_events():
            if event.type == pygame.QUIT:
                game.game_over = True
                episode = NUM_EPISODE
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = Action.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = Action.RIGHT
                elif event.key == pygame.K_UP:
                    action = Action.UP
                elif event.key == pygame.K_DOWN:
                    action = Action.DOWN

        game.step(action)
        game.render()          
    episode += 1
    game.reset()
game.quit()

'''
        
    '''