from DQN import DQN_Snake
from snake import Action
from snake import Snake_Game
from matplotlib import pyplot as plt
import pygame

PANNEL_HEIGHT = 80
WINDOW_WIDTH = 160
BLOCKSIZE = 20
TICK = 8
NUM_EPISODE = 1000

game = Snake_Game(PANNEL_HEIGHT, WINDOW_WIDTH, BLOCKSIZE, "human")

episode= 0
x_change = 0
y_change = 0

model = DQN_Snake(game.height_grid, game.width_grid, 4)


while episode < NUM_EPISODE:
    state = game.grid
    action = [0, 0]
    while not game.game_over: 
        action = model.select_action(game.grid)
        state, next_state, reward, done = game.step(list(Action)[action])
        model.memory.push(state, action.item(), next_state, reward, done)
        render = game.render()
    episode += 1
    if (episode % 5 == 0):
        print("training day")
        model.train_model()
    game.reset()
game.quit()


'''
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
'''