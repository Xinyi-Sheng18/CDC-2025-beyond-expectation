# Multi-step DQN training on LunarLander-v2
# Sampling strategy: Markovian sampling without overlap (e = 1)
# Based on: https://www.katnoria.com/nb_dqn_lunar/
# Modified with custom reward computation and multi-step return
import os
import random
import sys
from time import time
from collections import deque, defaultdict, namedtuple
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

"""Hyperparameters for DQN training"""

BUFFER_SIZE = int(1e5)  # Replay memory size
BATCH_SIZE = 64         # Number of experiences to sample from memory
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Soft update parameter for updating fixed q network
LR = 1e-4               # Q Network learning rate
UPDATE_EVERY = 4        # How often to update Q network

MAX_EPISODES = 2000  # Max number of episodes to play
MAX_STEPS = 2000     # Max steps allowed in a single episode/play
ENV_SOLVED = 200     # MAX score at which we consider environment to be solved
PRINT_EVERY = 100    # How often to print the progress

# Epsilon schedule
EPS_START = 1.0      # Default/starting value of eps
EPS_DECAY = 0.999    # Epsilon decay rate
EPS_MIN = 0.01       # Minimum epsilon

MULTI_STEP = 2       # Multi-step Q learning with step 2 

MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Use GPU (CUDA) if available, otherwise fallback to CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Build a fully connected neural network

        Parameters
        ----------
        state_size (int): State dimension
        action_size (int): Action dimension
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Replay memory allow agent to record experiences and learn from them

        Parametes
        ---------
        buffer_size (int): maximum size of internal memory
        batch_size (int): sample size from experience
        seed (int): random seed
        """
        self.batch_size = batch_size  # 64
        self.memory = deque(maxlen=buffer_size) # 1e6
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sample randomly and return (state, action, reward, next_state, done) tuple as torch tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert to torch tensors
        states = torch.from_numpy(
            np.vstack([experience.state for experience in experiences if experience is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([experience.action for experience in experiences if experience is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(
            device)
        # Convert done from boolean to int
        dones = torch.from_numpy(
            np.vstack([experience.done for experience in experiences if experience is not None]).astype(
                np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        DQN Agent interacts with the environment,
        stores the experience and learns from it

        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        #self.seed = random.seed(seed)
        # Initialize Q and Fixed Q networks
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.fixed_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Initiliase memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        """
        Update Agent's knowledge

        Parameters
        ----------
        state (array_like): Current state of environment
        action (int): Action taken in current state
        reward (float): Reward received after taking action
        next_state (array_like): Next state returned by the environment after taking action
        done (bool): whether the episode ended after taking action
        """
        self.memory.add(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):
        """
        Learn from experience by training the q_network

        Parameters
        ----------
        experiences (array_like): List of experiences sampled from agent's memory
        """
        states, actions, rewards, next_states, dones = experiences
        # Get the action with max Q value
        action_values = self.fixed_network(next_states).detach()
        # Notes
        # tensor.max(1)[0] returns the values, tensor.max(1)[1] will return indices
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([64]) -> torch.Size([64, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # (updated) If done just use reward, else update Q_target with discounted action values 
        # origin: Q_target = rewards + (GAMMA * max_action_values * (1 - dones)), we updated with multi-step bootstrapping
        Q_target = rewards + (GAMMA** MULTI_STEP* max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)

    def update_fixed_network(self, q_network, fixed_network):
        """
        Update fixed network by copying weights from Q network using TAU param

        Parameters
        ----------
        q_network (PyTorch model): Q network
        fixed_network (PyTorch model): Fixed target network
        """
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)

    def act(self, state, eps=0.0):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        eps (float): epsilon for epsilon-greedy action selection
        """
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # set the network into evaluation mode
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Back to training mode
            self.q_network.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)


def modified_GM(x, n):
    """
    Compute the modified geometric mean
    """
    magnitude = abs(x) ** (1. / n)  
    if x >= 0: return magnitude
        
    return -1 * magnitude

def main(args):
    """
    Main training loop for multi-step DQN on LunarLander-v2.

    Uses a Markovian sampling strategy with buffer clearing ("without overlap").
    """
    seed = args.seed    # Seed for reproducibility
    print("Running Python script with seed:", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    env = gym.make('LunarLander-v2')
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # Use a fixed-length buffer to store multi-step samples without overlap
    Multi_Step_return_buffer = deque(maxlen=MULTI_STEP)  

    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('State size: {}, action size: {}'.format(state_size, action_size))
    print('Multi step size: {}, lambda: {}'.format(MULTI_STEP, LAMBDA))
    dqn_agent = DQNAgent(state_size, action_size)

    start = time()
    scores = []

    # Maintain a list of last 100 scores
    scores_window = deque(maxlen=100)
    eps = EPS_START
    for episode in tqdm(range(1, MAX_EPISODES + 1)):
        state = env.reset()[0]
        score = 0
        for t in range(MAX_STEPS):
            action = dqn_agent.act(state, eps)
            next_state, reward, done,_, info = env.step(action) 
            
            # (updated)
            # store multi step boost trapping
            Multi_Step_return_buffer.append((state, action, reward, done))

            if len(Multi_Step_return_buffer) >= MULTI_STEP or done:
                G,delta = 0,0
                for i in range(len(Multi_Step_return_buffer)):
                    G += (GAMMA** i) * Multi_Step_return_buffer[i][2]
                    delta += Multi_Step_return_buffer[i][2]
                G = (1-LAMBDA) * G + LAMBDA * modified_GM(delta, len(Multi_Step_return_buffer)) * (1-GAMMA ** len(Multi_Step_return_buffer))
                
                # Clear the buffer after computing and using the multi-step return
                state, action, _,_ = Multi_Step_return_buffer[0]
                Multi_Step_return_buffer.clear()
                dqn_agent.step(state, action, G, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                if score >= ENV_SOLVED:
                    mean_score = np.mean(scores_window)
                    print('\rEnvironment solved in {} episodes, average score: {:.2f}'.format(episode, mean_score), end="")
                    sys.stdout.flush()
                    dqn_agent.checkpoint(f'{MODEL_DIR}/DQN_solved_200_{MULTI_STEP}GM{LAMBDA}_seed{seed}.pth')
                break
            
            eps = max(eps * EPS_DECAY, EPS_MIN)
            
            if score >= ENV_SOLVED:
                mean_score = np.mean(scores_window)
                print('\rEnvironment solved in {} episodes, average score: {:.2f}'.format(episode, mean_score), end="")
                sys.stdout.flush()
                dqn_agent.checkpoint(f'{MODEL_DIR}/DQN_solved_200_{MULTI_STEP}GM{LAMBDA}_seed{seed}.pth')
                break

        if episode % PRINT_EVERY == 0:
            mean_score = np.mean(scores_window)
            print('Episode {}, score : {:.2f}'.format(episode, mean_score), end="")
        scores_window.append(score)
        scores.append(score)

    end = time()
    print('Took {} seconds'.format(end - start))
    np.save(f'{RESULT_DIR}/{MULTI_STEP}DQN{LAMBDA}_Training_seed{seed}.npy',scores)
    print( f'Median value: {np.median(scores)}' )
    print( f'mean value: {np.mean(scores)}' )


    plt.figure(figsize=(10,6))
    plt.plot(scores)
    
    plt.plot(pd.Series(scores).rolling(100).mean())
    plt.title(f'{RESULT_DIR}/{MULTI_STEP}DQN{LAMBDA}_Training')
    plt.xlabel('# of episodes')
    plt.ylabel('score')
    plt.show()
    plt.savefig(f"{RESULT_DIR}/{MULTI_STEP}DQN{LAMBDA}_Training_seed{seed}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2-step DQN with custom lambda and seed.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed value")
    parser.add_argument("--lambda_val", type=float, default=0.0, help="Lambda value for reward shaping (default: 0.0)")
    
    args = parser.parse_args()

    # Assign parsed lambda value to the global LAMBDA variable
    LAMBDA = args.lambda_val
    
    main(args)
