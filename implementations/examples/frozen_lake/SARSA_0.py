import sys
sys.path.append('../../') 
import algorithms.TD.SARSA_0_on_policy_TD as TDiter 
import gym
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns

ENV_NAME = 'FrozenLake-v0'

def play_game_with_policies(env_name, list_of_Qs, output_file_path):
    """This function takes an environment, and a list of policies
    which it wants to compare performance between on this environment."""
    iteration_num = 0
    env = gym.make(env_name)
    policy_average_rewards = []
    policy_names=[]
    for Q_func in list_of_Qs:
        policy_name = 'iteration_'+str(iteration_num)
        policy_names.append(policy_name)
        rewards_by_episode = []
        for episode in range(500):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                next_action = np.argmax(Q_func[state])
                state, reward, done, info = env.step(next_action) 
                episode_reward += reward
            rewards_by_episode.append(episode_reward)
        iteration_num+=1
        policy_average_rewards.append(mean(rewards_by_episode))
    plt.title('Average Reward by Policy')
    bars = plt.bar(policy_names, policy_average_rewards)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.savefig(fname=output_file_path)

if __name__ == '__main__':

    # Create your environment
    env = gym.make(ENV_NAME)
    env.reset()

    # Create policy iterator object
    policy_iterator = TDiter.SARSA_0_on_policy(env=env,
                                          discount_factor=1,
                                          learning_rate=0.15)


    # Call iterate function and get all the policies iterated through
    list_of_Qs = policy_iterator.iterate(iteration_loops=500001,
                                         print_every_n=50000,
                                         eval_episodes=1,
                                         epsilon_soft=0.05)
    print('''Check these Qs to make sure they are sufficiently different!
{}'''.format(list_of_Qs))

    # Play game with generated policies and save results!
    play_game_with_policies(env_name=ENV_NAME,
                            list_of_Qs=list_of_Qs,
                            output_file_path='outputs/SARSA_0.png')

