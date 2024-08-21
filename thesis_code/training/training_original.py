import os
import torch
import gym_super_mario_bros
import random
import csv
from gym_super_mario_bros.actions import RIGHT_ONLY
from agent import MarioAgent
from nes_py.wrappers import JoypadSpace
from ..util.wrappers import apply_wrappers

if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
else:
    print("no CUDA")

CHECKPOINT_INTERVAL = 2500
EPISODES = 30000
LOAD_MODEL = False

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

mario_agent = MarioAgent(input_shape=env.observation_space.shape, num_actions=env.action_space.n)

os.makedirs("models", exist_ok=True)

if LOAD_MODEL:
    mario_agent.load_model(os.path.join("models", "model_original_15000.pt"))
    print("Loaded model")

# keep track of reward over time
csv_file = 'rewards_2.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Reward', 'Steps', 'Exploration Rate'])

# Learning loop
env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
for i in range(EPISODES):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        action = mario_agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        mario_agent.store_in_memory(state, action, reward, new_state, done)
        mario_agent.learn()
        state = new_state
    print("Total reward:", total_reward, "Exploration rate:", mario_agent.exploration_rate, "Replay buffer size:", len(mario_agent.replay_buffer), "Step counter:", mario_agent.steps)

    # Save episode reward to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, total_reward, mario_agent.steps, mario_agent.exploration_rate])

    if CHECKPOINT_INTERVAL > 0 and (i + 1) % CHECKPOINT_INTERVAL == 0:
        mario_agent.save_model(os.path.join("models", "model_original_" + str(i + 1) + ".pt"))

mario_agent.save_model(os.path.join("models", "model_original_" + str(EPISODES) + ".pt"))
env.close()
