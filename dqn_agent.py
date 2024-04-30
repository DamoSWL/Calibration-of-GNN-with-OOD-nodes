import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
import torch.nn.functional as F
import math
from pathlib import Path
import logging
import scipy.sparse as sp

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class Memory(object):
    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):
        samples = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*samples))
        
        state_batch = torch.vstack(batch.state)     
        action_batch = torch.vstack(batch.action).view(-1)
        reward_batch = torch.vstack(batch.reward).view(-1)
        next_state_batch = torch.vstack(batch.next_state)
        done_batch = torch.vstack(batch.done).view(-1)
   
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def __len__(self):
        return len(self.memory)



class DQNAgent(object):
    def __init__(self,state_shape=None,memory_size=1e4,dataset='cora',model_path='cora_HyperU_RL',weight=0.5):

        self.replay_memory_size = memory_size
        self.discount_factor = 0.95
        self.batch_size = 64
        self.action_num = 2
        self.learning_rate=0.0005
        self.dataset = dataset
        self.model_path = model_path

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.memory = Memory(self.replay_memory_size, self.batch_size)

        self.q_estimator = QNetworkProxy(action_num=self.action_num, learning_rate=self.learning_rate, state_shape=state_shape, \
            device=self.device)
        self.target_estimator = QNetworkProxy(action_num=self.action_num, learning_rate=self.learning_rate, state_shape=state_shape, \
            device=self.device)

        self.tau = 0.001

        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay= None
        self.eps_cnt = 0
        self.target_update = 1

        self.weight = weight


    def feed_memory(self, state, action, reward, next_state, done):
        self.memory.save(state, action, reward, next_state, done)

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state, action, reward, next_state, done)
       
    def select_action(self,state):
        if len(self.memory) < self.batch_size:
            return torch.randint(2,(state.size(0),1),device=self.device).view(-1)
        else:
            sample = np.random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.eps_cnt / self.eps_decay)
            self.eps_cnt += 1
            if sample > eps_threshold:
                return self.step(state)
            else:
                return torch.randint(2,(state.size(0),1),device=self.device).view(-1)

    
    def learn(self, env, total_timesteps):
        self.eps_decay = total_timesteps

        state = env.reset()
        snaps = []
   
        for t in range(total_timesteps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            snaps = zip(state, action, reward, next_state, done)
            for each in snaps:
                self.feed(each)

            self.train(t)
            state = next_state



    def train(self,t):
        if len(self.memory) < 5 * self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample()

        q_values_next_target = self.target_estimator.predict(next_state).max(dim=-1)[0]

        reward = reward.to(self.device)
        done = done.to(self.device)
        target_batch = reward + (1-done.int()) *self.discount_factor * q_values_next_target

        self.q_estimator.update(state, action, target_batch)

        if t % self.target_update == 0:
            target_net_state_dict = self.target_estimator.qnet.state_dict()
            policy_net_state_dict = self.q_estimator.qnet.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_estimator.qnet.load_state_dict(target_net_state_dict)



    def step(self, states):
        q_values = self.q_estimator.predict(states)
        best_actions = q_values.max(dim=-1)[1].view(-1)
        return best_actions



    def step_adj_weight(self, states):
        q_values = self.q_estimator.predict(states)
        best_actions = q_values.max(dim=-1)[1].view(-1)

        adj_weight = np.ones(states.size(0))
        mask = best_actions.cpu().numpy()<1
        adj_weight[mask] = self.weight

        return adj_weight.tolist()


    def save_model(self):
        path =Path(self.model_path)
        if not path.exists():
            path.mkdir()
        logging.info(f'save the Q network')
        torch.save(self.q_estimator.qnet.state_dict(),str(path / 'policy_best_model.pth'))
        torch.save(self.target_estimator.qnet.state_dict(),str(path / 'target_best_model.pth'))



    def load_model(self):
        path =Path(self.model_path)
        if path.exists():
            self.q_estimator.qnet.load_state_dict(torch.load(str(path / 'policy_best_model.pth')))
            self.target_estimator.qnet.load_state_dict(torch.load(str(path / 'target_best_model.pth')))


class QNetworkProxy(object):
    def __init__(self, state_shape=1000, action_num=2, learning_rate=0.001, device=None):

        self.device = device

        # set up Q model and place it in eval mode
        self.qnet = QNetwork(state_shape,action_num)
        self.qnet = self.qnet.to(self.device)
        self.qnet.eval()

        self.mse_loss = nn.MSELoss(reduction='mean')
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

    def predict(self, s):
        self.qnet.eval()
        with torch.no_grad():
            s = s.to(self.device)
            q_as = self.qnet(s)
        return q_as

    def update(self, s, a, y):
        self.optimizer.zero_grad()

        self.qnet.train()

        s = s.to(self.device).float()
        a = a.to(self.device).long()
        y = y.to(self.device).float()

        q_as = self.qnet(s)
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).view(-1)


        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        

        self.qnet.eval()

        return batch_loss


class QNetwork(nn.Module):
    def __init__(self,state_shape,action_num=2):

        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(state_shape, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, action_num)



    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)
        