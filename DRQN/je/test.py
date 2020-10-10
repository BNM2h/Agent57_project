import gym
import numpy as np
env=gym.make('Pong-v0')

obs=env.reset()

for i in range(1000):
    env.render()
    obs,reward,done,info=env.step(env.action_space.sample())
    print(env.action_space.sample())
    #print(obs.shape) 210*160*3
