import os
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import MarioAgent
from nes_py.wrappers import JoypadSpace
from ..util.wrappers import apply_wrappers
import csv

if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
else:
    print("no CUDA")
UNSEEN_LEVEL_3 = 'SuperMarioBros-1-1-v1'
UNSEEN_LEVEL_2 = 'SuperMarioBros-1-1-v3'
UNSEEN_LEVEL_1 = 'SuperMarioBros-1-1-v2'
SHOW_GAME = False
EPISODES = 1000

def test_agent(agent, env, episodes):
    sum_of_rewards = 0
    for i in range(episodes):
        print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = new_state
        sum_of_rewards += total_reward
    average_reward = sum_of_rewards / episodes
    return average_reward

env = gym_super_mario_bros.make(UNSEEN_LEVEL_3, render_mode='human' if SHOW_GAME else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# testing the first agent
agent1 = MarioAgent(input_shape=env.observation_space.shape, num_actions=env.action_space.n)
agent1_model = "model_original_30000.pt"
agent1.load_model(os.path.join("models", agent1_model))
agent1.exploration_rate = 0.1
agent1.exploration_decay = 0.0
agent1.exploration_min = 0.0

print("Testing Agent 1")
average_reward_agent1 = test_agent(agent1, env, EPISODES)

# testing the second agent
agent2 = MarioAgent(input_shape=env.observation_space.shape, num_actions=env.action_space.n)
agent2_model = "model_custom_30000.pt"
agent2.load_model(os.path.join("models", agent2_model))
agent2.exploration_rate = 0.1
agent2.exploration_decay = 0.0
agent2.exploration_min = 0.0

print("Testing Agent 2")
average_reward_agent2 = test_agent(agent2, env, EPISODES)

print(f"Average reward for Agent 1: {average_reward_agent1}")
print(f"Average reward for Agent 2: {average_reward_agent2}")
