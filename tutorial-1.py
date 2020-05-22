import gym
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


env = gym.make('MountainCar-v0')


LEARNING_RATE = 0.1
logger.info('Setting learning rate to {}'.format(LEARNING_RATE))
DISCOUNT = 0.95
logger.info('Discount factor is {}'.format(DISCOUNT))
EPISODES = 50*10**3
logger.info("Setting episodes to {}".format(EPISODES))
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discerete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

logger.info('discrete_os_win_size: {}'.format(discerete_os_win_size))

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [(env.action_space.n)]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discerete_os_win_size
    return tuple(discrete_state.astype(np.int))





for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        logger.info('Episode : {}'.format(episode))
        render = True
    else:
        render = False
    done = False
    discrete_state = get_discrete_state(env.reset())
    logger.info('Starting discrete state is {}'.format(discrete_state))
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _  = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

env.close()