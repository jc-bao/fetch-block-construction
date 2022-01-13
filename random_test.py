import gym
import numpy as np
from gym.wrappers.monitor import Monitor
import fetch_block_construction
env = gym.make('FetchTower-v0')
# env = Monitor(env, directory="videos", force=True, video_callable=lambda x: x)
env.reset()
step=0
num_obj = 1
while True:
    obs, done =env.reset(), False
    while not done:
        # env.render()
        action = env.action_space.sample()
        step_results = env.step(action)
        obs, reward, done, info = step_results
        if step==env._max_episode_steps:
            print(env._max_episode_steps)
            step = 0
            # num_obj+=1
            env.change(num_obj)
            env.reset()
        step+=1
        env.render()