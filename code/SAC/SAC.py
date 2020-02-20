"""
Soft Actor Critic
entropy temperature
double Q learning
delayed Q learning

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal

import gym
import random
import numpy as np
from collections import deque

from pendulum_2 import PendulumEnv as penv 
#from buffers.buffer import PrioritizedReplayBuffer

import matplotlib.pyplot as plt
from matplotlib import animation

from tensorboardX import SummaryWriter

import pdb

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Makes use of a SumTree to find prioritized replay
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, error, sample):
    #def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = batch[0]
        actions = batch[1] 
        rewards = batch[2] 
        next_states = batch[3] 
        dones = batch[4] 

        return states, actions, rewards, next_states, dones, idxs, is_weight
        #return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

def plot(frame_idx, rewards):
        #clear_output(True)
        plt.figure(figsize=(5,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
        plt.plot(rewards)
        plt.show()

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    update_step = 0

    writer = SummaryWriter(comment='Test 1', logdir="~/board_data/")

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)#NOTE
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            #! Push sample to agent to calculate error prior to pushing to buffer
            #agent.push(state, action, reward, next_state, done)

            episode_reward += reward
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)#NOTE
                update_step += 1

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                break

            state = next_state

        writer.add_scalar("reward2", episode_reward, episode)
        
        print("Episode " + str(episode) + ": " + str(episode_reward))
    writer.close()

    return episode_rewards


class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        """[summary]
        
        Arguments:
            batch_size {[type]} -- [description]
        
        Returns:
            list of batch_size np.arrays
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)

    
class ValueNetwork(nn.Module):

    #def __init__(self, input_dim, output_dim, init_w=3e-3):
    def __init__(self, state_dim, output_dim=1, hidden_dim=256, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.linear3 .weight.data.uniform_(-init_w, init_w)
        self.linear3 .bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        #! NOTE  this is where multi input magic happens
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)


        # Mean and Std of Gaussian Distribution for actions
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):#NOTE
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi


class SACAgent:
  
    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        # initialize networks 
        self.q_net1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.q_net2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.target_q_net1 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.target_q_net2 = SoftQNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        # initialize optimizers mean
        self.q1_optimizer       = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer       = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer   = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = ReplayBuffer(buffer_maxlen)
        #self.replay_buffer = PrioritizedReplayBuffer(buffer_maxlen)

    def get_action(self, state):#NOTE
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (self.action_range[1] + self.action_range[0]) / 2.0

    def push(self, state, action, reward, next_state, done):
        #target = torch.min(self.q_net1, self.q_net2)
        """
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(np.array([reward])).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(np.array([done])).to(self.device)
        done = done.view(done.size(0), -1)

        
        pdb.set_trace()
        target = self.q_net2(Variable(torch.FloatTensor(state)), Variable(torch.FloatTensor(action))).data
        old_val = target[0][action]
        target_val = self.target_q_net2(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])
        """
        error = 1
        self.replay_buffer.push(error, (state, action, reward, next_state, done))
        

    def update(self, batch_size):

        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        #states, actions, rewards, next_states, dones, idxs, is_weight = self.replay_buffer.sample(batch_size)

        pdb.set_trace()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states) #NOTE
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        print('update')
        pdb.set_trace()
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)        
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        #! Attempt to do PER
        #! ---------------------------------------------------------------
        pdb.set_trace()
        predicted_state = expected_q
        actual_next_state = next_states

        # | Q_t - Q_t-1 |
        errors = torch.abs(predicted_state - actual_next_state).data.numpy()

        q_loss_min = Variable(torch.min(q1_loss, q2_loss))
        for i in range(batch_size):
            self.replay_buffer.update(idxs[i], errors[i])

        loss = (torch.FloatTensor(is_weights) * q_loss_min).mean()
        loss.backward()
        #! ---------------------------------------------------------------

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - mini_batch_train).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # target networks
            for target_param, param in zip(self.tmini_batch_train_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1



if __name__ == "__main__":
    #env = gym.make("Pendulum-v0")
    #env = gym.make("LunarLander-v2")
    env = penv()

    # SAC 2019 Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 3e-4
    q_lr = 3e-4
    p_lr = 3e-4
    buffer_maxlen = 1000000

    # 2019 agent
    agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

    # train
    episode_rewards = mini_batch_train(env, agent, 100, 500, 64)


    env = gym.make("Pendulum-v0")
    print('Testing: ')

    # Run a demo of the environment
    state = env.reset()
    cum_reward = 0
    frames = []
    #for t in range(5000000):
    while True:
        # Render into buffer. 
        frames.append(env.render(mode = 'rgb_array'))
        action = agent.get_action(state)
        state, reward, done, info = env.step(action)
    #    if done:
    #        break
    env.close()

    print(np.mean(episode_rewards))

