from DQN3 import DQN_Snake
from snake import Snake_Game
from matplotlib import pyplot as plt
import pygame

PANNEL_HEIGHT = 0
WINDOW_WIDTH = 400
BLOCKSIZE = 20
TICK = 8
NUM_EPISODE = 1000

game = Snake_Game(PANNEL_HEIGHT, WINDOW_WIDTH, BLOCKSIZE, "rgb_array")

episode= 0
x_change = 0
y_change = 0    

model = DQN_Snake(game.height_grid, game.width_grid, 4, game)


max_eps = 1
min_eps = 0.01 
eps_decay = 200

rewards = [] 
best_score = 0

while episode < NUM_EPISODE:
    game.reset()
    iteration = 0
    while not game.game_over:
        epsilon = max(max_eps - episode / eps_decay, min_eps)
        render = model.env.render()
        next_state, reward, done, score = model.play_one_step(render, epsilon)

        #plt.imshow(render)
        #plt.show()
        iteration += 1
        if done:
            break
    rewards.append(score)
    if score >= best_score: # Not shown
        #best_weights = model.weights() # Not shown
        best_score = score # Not shown
    print("\rEpisode: {:4}, Steps: {:3}, eps: {:.3f}, score: {:3}".format(episode, iteration + 1, epsilon, score), end="") # Not shown
    if episode > 10:
        model.training_step()
    episode += 1
print("\nBest Score:", best_score)
'''while episode < NUM_EPISODE:
    state = game.grid 
    action = [0, 0]
    while not game.game_over: 
        action = model.select_action(game.grid)
        state, next_state, reward, done = game.step(list(Action)[action])
        model.memory.push(state, action.item(), next_state, reward, done)
        render = game.render()
        #plt.imshow(render)
        #plt.show()
    episode += 1
    if (episode % 10 == 0):
        print("training day")
        model.train_model()
    game.reset()
game.quit()
'''
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