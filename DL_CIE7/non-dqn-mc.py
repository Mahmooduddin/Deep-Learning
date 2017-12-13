# -*- coding: utf-8 -*-
import random

import gym
import numpy as np

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            next_state, reward, done, _ = env.step(random.randrange(3))
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                break

        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
