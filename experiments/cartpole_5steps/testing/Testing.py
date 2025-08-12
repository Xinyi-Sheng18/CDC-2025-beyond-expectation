import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import argparse
import os


NUM_BINS = 10
N_STEP = 5


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


def test_agent(env, policy, state_space_bounds, episodes=100):
    """
    Evaluate a trained policy on the environment
    """
    rewards = []
    t200, t500 = 0, 0

    for epi in range(episodes):
        state = discretize_state(env.reset()[0], state_space_bounds)
        done = False
        total_reward = 0
        t = 0

        while not done:
            action = np.argmax(policy[state])
            next_state_raw, reward, done, truncated, _ = env.step(action)
            state = discretize_state(next_state_raw, state_space_bounds)
            total_reward += reward
            t += 1

            if truncated:
                t500 += 1
                break
        
        if t >= 200:
            t200 += 1
        
        rewards.append(total_reward)

    return rewards, t200, t500


def main(args):
    seed = args.seed
    lambda_val = args.lambda_val
    print(f"Running testing with seed={seed}, lambda={lambda_val}")

    random.seed(seed)
    np.random.seed(seed)

    # Initialize environment
    env = gym.make("CartPole-v1")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # State bounds for discretization
    state_space_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_space_bounds[1] = [-0.5, 0.5]    # Velocity bound
    state_space_bounds[3] = [-math.radians(50), math.radians(50)]  # Angular velocity bound

    # Load learned policy and training reward
    q_table_path = f"../training/models/Q_table_{N_STEP}step{lambda_val}_seed{seed}.npy"
    #reward_path = f"../rewards/Training_reward_{N_STEP}step{lambda_val}_seed{seed}.npy"
    q_table = np.load(q_table_path)
    #training_rewards = np.load(reward_path)

    # Evaluate the policy
    test_rewards, t200, t500 = test_agent(env, q_table, state_space_bounds, episodes=100)

    # Save testing results
    os.makedirs("testing_results", exist_ok=True)
    test_save_path = f"testing_results/Testing_result_{N_STEP}GM{lambda_val}_seed{seed}.npy"
    np.save(test_save_path, test_rewards)

    # Print statistics
    print(f"GM_N{N_STEP}, λ={lambda_val}, seed={seed}")
    print(f"Rewards > 200: {t200} times")
    print(f"Rewards > 500 (truncated): {t500} times")
    #print(f"Training  Median: {np.median(training_rewards):.2f}, Mean: {np.mean(training_rewards):.2f}")
    print(f"Testing   Median: {np.median(test_rewards):.2f}, Mean: {np.mean(test_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run testing script.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed value")
    parser.add_argument("--lambda_val", type=float, default=0.0, help="Lambda value used in training")
    args = parser.parse_args()
    main(args)
