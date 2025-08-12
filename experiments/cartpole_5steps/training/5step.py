import numpy as np
import gym
import random
import math
import matplotlib.pyplot as plt
from collections import deque  
import argparse
from tqdm import tqdm
import os 



# Hyperparameters
LR = 0.1                # Learning rate
GAMMA = 0.99            # Discount factor
EPSILON_START = 1.0           # Starting value for epsilon
EPSILON_DECAY = 0.995   # Decay rate for epsilon
EPSILON_MIN = 0.01      # Minimum value for epsilon
EPISODES = 100_000      # Total training episodes
N_STEP = 5              # Multi-step size 
NUM_BINS = 10           # Number of bins for state discretization


def discretize_state(state, state_space_bounds):
    """
    Discretize continuous state into bins
    """
    discretized = []
    for i in range(len(state)):
        if state[i] <= state_space_bounds[i][0]:
            discretized.append(0)
        elif state[i] >= state_space_bounds[i][1]:
            discretized.append(NUM_BINS - 1)
        else:
            scale = (state[i] - state_space_bounds[i][0]) / (state_space_bounds[i][1] - state_space_bounds[i][0])
            discretized.append(int(scale * NUM_BINS))
    return tuple(discretized)


def Modified_GM(x, n):
    """
    Compute the modified geometric mean
    """
    magnitude = abs(x) ** (1. / n)  
    if x >= 0: return magnitude
    return -1 * magnitude


class MQAgent:
    """
    Multi-step Q-learning agent with Modified Geometric Mean (GM) reward shaping.
    """

    def __init__(self, env, lambda_coeff, seed):
        self.env = env
        self.lambda_coeff = lambda_coeff
        self.seed = seed
        self.epsilon = EPSILON_START
        self.q_table = np.zeros((NUM_BINS,) * env.observation_space.shape[0] + (env.action_space.n,))
        self.rewards = []

        random.seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        self.n_actions = env.action_space.n
        self.state_space_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_space_bounds[1] = [-0.5, 0.5]    # Velocity bound
        self.state_space_bounds[3] = [-math.radians(50), math.radians(50)]  # Angular velocity bound


    def train(self):
        for episode in tqdm(range(EPISODES)):
            state = discretize_state(self.env.reset()[0],self.state_space_bounds)
            done = False
            total_reward = 0
            buffer = deque(maxlen=N_STEP)
            
            while not done:
                # epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])
                
                next_state_conti, reward, done, truncated, _ = self.env.step(action)
                next_state = discretize_state(next_state_conti, self.state_space_bounds)
                buffer.append((state, action, reward))

                # Update Q-table if enough experience is collected
                if len(buffer) == N_STEP:
                    # 计算 n-step return
                    delta = Modified_GM(sum( buffer[i][2] for i in range(N_STEP)  ), N_STEP)
                    G = sum([GAMMA ** i * buffer[i][2] for i in range(N_STEP)])
                    G = (1 - self.lambda_coeff) * G + self.lambda_coeff * delta * (1-GAMMA**N_STEP)

                    # truncated : success, over 500. done: failed
                    if not done:
                        G += GAMMA ** N_STEP * np.max(self.q_table[next_state])
                    
                    s_tau, a_tau, _ = buffer.popleft()
                    self.q_table[s_tau][a_tau] += LR * (G - self.q_table[s_tau][a_tau])
                
                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            # Handle remaining transitions in buffer
            while  len(buffer) > 0: 
                G = sum([GAMMA ** i * buffer[i][2] for i in range(len(buffer))])
                delta = Modified_GM(sum( buffer[i][2] for i in range(len(buffer)) ), N_STEP) #len(buffer)
                G = ((1 - self.lambda_coeff) *
                            G + self.lambda_coeff * delta * (1-GAMMA**N_STEP) ) # 加上未来状态的 Q 值
                s_tau, a_tau, _ = buffer.popleft()
                self.q_table[s_tau][a_tau] += LR * (G - self.q_table[s_tau][a_tau])
            
            self.rewards.append(total_reward)

            # Decay epsilon every 100 episodes
            if episode % 100 == 0 and self.epsilon > EPSILON_MIN:
                self.epsilon *= EPSILON_DECAY

            # Print results every 1000 episode
            if episode % 1000 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

        self.env.close()

    def save_results(self):
        """
        Save the learned Q_table as the indicator of action selection
        """
        os.makedirs("models", exist_ok=True)
        os.makedirs("rewards", exist_ok=True)
        np.save(f"models/Q_table_{N_STEP}step{self.lambda_coeff}_seed{self.seed}.npy", self.q_table)
        np.save(f"rewards/Training_reward_{N_STEP}step{self.lambda_coeff}_seed{self.seed}.npy", self.rewards)

        """
        # Plot the cumulative reward after episodes
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'GM {N_STEP}-step GM{lambda_coeff} on CartPole_seed{seed} ')
        plt.show()
        plt.savefig(f"{N_STEP}step{lambda_coeff}_Training_seed{seed}.png")
        """



def main(args):
    # Set seeds for reproducibility
    seed = args.seed
    print("Running Python script with seed:", seed)

    env = gym.make('CartPole-v1')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    agent = MQAgent(env, lambda_coeff=args.lambda_val, seed=seed)
    agent.train()
    agent.save_results()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the script with a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed value")
    parser.add_argument("--lambda_val", type=float, default=0.0, help="Lambda value for reward shaping (default: 0.0)")
    args = parser.parse_args()
    
    main(args)

