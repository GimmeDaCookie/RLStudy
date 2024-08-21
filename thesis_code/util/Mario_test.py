import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

EPISODES = 100
env = gym_super_mario_bros.make('SuperMarioBros-v1', render_mode = 'human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

for i in range(EPISODES):    
    print("Episode:", i)
    env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        _, _, done, _, _ = env.step(action)
        env.render()