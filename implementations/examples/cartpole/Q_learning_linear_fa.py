import sys
sys.path.append('../../') 
import algorithms.LFA.Q_learning_LFA as Q_LFA
import logging
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean

logging.basicConfig(level=logging.INFO)

def play_game_with_LFA(env_name, LFAs, output_file_path):
    """This function takes an environment, and a series of LFA policies
    which it wants to compare performance between on this environment."""
    iteration_num = 0
    env = gym.make(env_name)
    policy_average_rewards = []
    policy_names=[]
    for LFA in LFAs:
        policy_name = 'iteration_'+str(iteration_num)
        logging.info('Testing {}'.format(policy_name))
        logging.info('LFA.weights:\n{}'.format(LFA.weights))
        policy_names.append(policy_name)
        rewards_by_episode = []
        for episode in range(100):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                next_action = np.argmax(LFA.predict(features_array=state))
                state, reward, done, info = env.step(next_action) 
                episode_reward += reward
            rewards_by_episode.append(episode_reward)
        iteration_num+=1
        policy_average_rewards.append(mean(rewards_by_episode))
        logging.info('policy_average_rewards: {}'.format(policy_average_rewards))
    plt.title('Average Reward by Policy')
    bars = plt.bar(policy_names, policy_average_rewards)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.savefig(fname=output_file_path)

if __name__ == '__main__':

    ENV_NAME = 'CartPole-v0'
    TRAINING_EPISODES = 10
    STEPS_PER_UPDATE = 5
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.998
    EPSILON = 0.3
    PRINT_EVERY_N_EPISODES = 1
    # Create your environment
    env = gym.make(ENV_NAME)
    env.reset()

    # Create Q_learner
    Q_learner_iterator = Q_LFA.Q_learner(env=env)

    # Call iterate function
    LFA_list = Q_learner_iterator.iterate(
            training_episodes=TRAINING_EPISODES,
            steps_per_update=STEPS_PER_UPDATE,
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            epsilon=EPSILON,
            print_every_n_episodes=PRINT_EVERY_N_EPISODES)

    # Play game with generated policies and save results!
    play_game_with_LFA(env_name=ENV_NAME,
                            LFAs=LFA_list,
                            output_file_path='outputs/LFA.png')

