from numpy import dtype
import torch
import random
import math
from collections import namedtuple, deque
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        #self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        #self.bn3 = torch.nn.BatchNorm2d(32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        #convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        #convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))

        linear_input_size = convw * convh * 32
        self.head = torch.nn.Linear(linear_input_size * 16, outputs)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        #x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x

class DQN_Snake:

    BATCH_SIZE = 256
    GAMMA = 0.999
    EPS_START = 0.95
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    LEARNING_RATE = 0.9

    def __init__(self, height, width, n_actions):
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_duration = []
        self.dqn = DQN(height, width, n_actions).to(self.device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr= self.LEARNING_RATE)
        self.memory = ReplayMemory(10000)
        self.step_done = 0

    def save(self):
        torch.save(self.optimizer.state_dict(), "./model.pt")

    def select_action(self, state):
        sample = random.random()
        state = torch.unsqueeze(state, 0)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.step_done / self.EPS_DECAY)
        self.step_done += 1
        if sample > eps_threshold:
            #return torch.argmax(self.dqn(state))
            return torch.tensor([[torch.argmax(self.dqn(torch.unsqueeze(state, 0)))]], device=self.device, dtype=torch.int32)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.int32)
            #return torch.tensor(random.randrange(self.n_actions))

    def train_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            print("Missing some state to start training")
            return
        print("Start training")
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=bool)

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        #with torch.no_grad():
        non_final_next_states = torch.stack(batch.next_state)
            
        #state_batch = batch.state #[[256][21][21]]
        #state_batch = np.array(list(batch.state), dtype=np.int32)
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)

        

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_batch = torch.unsqueeze(state_batch, 1)
        action_batch = torch.unsqueeze(action_batch, 1)
        non_final_next_states = torch.unsqueeze(non_final_next_states, 1)

        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE).type(torch.Tensor)
        next_state_values[non_final_mask] = self.dqn(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch #[n_batch][curennt_pred][prev_pred]
        expected_state_action_values = torch.unsqueeze(expected_state_action_values, 1)

        print("shape input : ", state_action_values.shape)
        print("shape target : ", expected_state_action_values.shape)

        # Compute Huber loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.save()

       
