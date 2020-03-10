import sys
sys.path.append('../../') 
import algorithms.dynamic_programming.policy_iteration as pi 
import gym
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

ENV_NAME = 'FrozenLake8x8-v0'

def play_game_with_policies(env_name, list_of_policies, output_file_path):
    """FrozenLake-v0 is considered "solved" when the agent obtains an average
    reward of at least 0.78 over 100 consecutive episodes."""
    iteration_num = 0
    lake_env = gym.make(env_name)
    policy_average_rewards = []
    policy_names=[]
    for policy in list_of_policies:
        policy_name = 'iteration_'+str(iteration_num)
        policy_names.append(policy_name)
        rewards_by_episode = []
        for episode in range(100):
            state = lake_env.reset()
            done = False
            episode_reward = 0
            while not done:
                next_action = np.argmax(policy[state,])
                state, reward, done, info = lake_env.step(next_action) 
                episode_reward += reward
            rewards_by_episode.append(episode_reward)
        plt.subplot(len(list_of_policies)+1, 1, iteration_num+1)
        plt.plot(rewards_by_episode)
        plt.title('{} Rewards Over Time'.format(policy_name))
        iteration_num+=1
        policy_average_rewards.append(mean(rewards_by_episode))
    plt.subplot(len(list_of_policies)+1, 1, len(list_of_policies)+1)
    plt.title('Average Over Episodes by Policy')
    bars = plt.bar(policy_names, policy_average_rewards)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.savefig(fname=output_file_path)

if __name__ == '__main__':

    lake_env = gym.make(ENV_NAME)
    lake_env.reset()

    policy_iterator = pi.policy_iterator(env=lake_env.env,
                                        evaluation_loops=10000,
                                        theta=0.0001,
                                        discount_factor=1)

    list_of_policies = policy_iterator.iterate(iteration_loops=500)

    play_game_with_policies(env_name=ENV_NAME,
                            list_of_policies=list_of_policies,
                            output_file_path='outputs/policy_comparison.png')
