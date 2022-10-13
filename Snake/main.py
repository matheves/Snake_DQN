from DQN import DQN_Snake
from env import Action
# from snake import Snake_Game
from matplotlib import pyplot as plt
import env



PANNEL_HEIGHT = 100
WINDOW_WIDTH = 400
BLOCKSIZE = 20
TICK = 8
NUM_EPISODE = 1000000
MODE = "Training" # Training or Eval

#game = Snake_Game(PANNEL_HEIGHT, WINDOW_WIDTH, BLOCKSIZE, "rgb_array")
game = env.Env(20)
episode= 0
x_change = 0
y_change = 0
score = []
epoch = 1
file = open("result.txt", "a")

model = DQN_Snake(game.size, game.size, 4)
model.load_model("./sam_model.pt")
model.load_optimizer("./sam_optimizer.pt")

if (MODE == "Training"):
    model.dqn.train()
else :
    model.dqn.eval()


while episode < NUM_EPISODE:
    state = game.grid
    action = [0, 0]
    while not game.game_over: 
        action = model.select_action(game.grid)
        state, next_state, reward, done = game.step(list(Action)[action])
        model.memory.push(state, action.item(), next_state, reward, done)
        #render = game.render()
    model.train_model()
    episode += 1
    score.append(game.score)
    if (episode % 5000 == 0):
        model.save_model()
        model.save_optimizer()
        result = "epoch : {} mean score : {} max score : {} \n".format(epoch, sum(score) / len(score), max(score))
        print(result)
        file.write(result)
        epoch += 1
        score = []
    game.reset()
file.close()
#game.quit()


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