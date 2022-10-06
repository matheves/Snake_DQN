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
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = torch.nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.nn.functional.relu(self.bn1(torch.unsqueeze(self.conv1(x), 0)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        #return self.head(x.view(x.size(0), -1))
        return self.head(x)

class DQN_Snake:

    BATCH_SIZE = 256
    GAMMA = 0.999
    EPS_START = 1
    EPS_END = 0.5
    EPS_DECAY = 1000
    TARGET_UPDATE = 10
    LEARNING_RATE = 0.5

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
            return torch.tensor([[torch.argmax(self.dqn(state))]], device=self.device, dtype=torch.int32)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.int32)
            #return torch.tensor(random.randrange(self.n_actions))

    def train_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        with torch.no_grad():
            non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                        volatile=True)
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = self.dqn(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(torch.Tensor))
            next_state_values[non_final_mask] = self.dqn(non_final_next_states).max(1)[0]
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            # Compute Huber loss
            loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.save()

        '''
        self.optimizer.zero_grad()

        targets = batch.reward + torch.mul((self.gamma * self.dqn(batch.next_state).max(1).values.unsqueeze(1)), 1 - batch.done)
        current = self.dqn(batch.state).gather(1, batch.action.long())

        
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
            if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        '''
