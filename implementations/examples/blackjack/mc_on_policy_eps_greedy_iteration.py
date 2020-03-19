import sys
sys.path.append('../../') 
import algorithms.MC.MC_on_policy_control_eps_greedy as MCiter 
import gym
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import seaborn as sns

ENV_NAME = 'Blackjack-v0'
# ENV_NAME = 'FrozenLake8x8-v0'

def play_game_with_policies(env_name, list_of_policies, output_file_path):
    """This function takes an environment, and a list of policies
    which it wants to compare performance between on this environment."""
    iteration_num = 0
    env = gym.make(env_name)
    policy_average_rewards = []
    policy_names=[]
    for policy in list_of_policies:
        policy_name = 'iteration_'+str(iteration_num)
        policy_names.append(policy_name)
        rewards_by_episode = []
        for episode in range(500):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                next_action = policy(state)
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

def plot_Q_func(Q_func, output_path):
    player_hands = [key[0] for key in Q_func.keys()]
    dealer_hands = [key[1] for key in Q_func.keys()]
    usable_ace = [key[2] for key in Q_func.keys()]
    choices = [np.argmax(value['Q(s,a)']) for value in Q_func.values()]
    Q_func_df = pd.DataFrame(data={'player':player_hands,'dealer':dealer_hands,'ace':usable_ace,'choices':choices})
    ace_df = Q_func_df[Q_func_df.ace==True] 
    no_ace_df = Q_func_df[Q_func_df.ace==False]
    ace_df = ace_df.pivot(index='player', columns='dealer', values='choices')
    no_ace_df = no_ace_df.pivot(index='player', columns='dealer', values='choices')
    ace_heatmap = sns.heatmap(data=ace_df, annot=True)
    ace_fig = ace_heatmap.get_figure()
    ace_fig.savefig(output_path+'_ace.png')
    no_ace_heatmap = sns.heatmap(data=no_ace_df, annot=True)
    no_ace_fig = no_ace_heatmap.get_figure()
    no_ace_fig.savefig(output_path+'_no_ace.png')
    # print(Q_func_df)


if __name__ == '__main__':

    # Create your environment
    env = gym.make(ENV_NAME)
    env.reset()

    # Create policy iterator object
    policy_iterator = MCiter.MC_on_policy(env=env,
                                          eval_episodes=1,
                                          discount_factor=1)

    # Call iterate function and get all the policies iterated through
    list_of_policies, list_of_Qs = policy_iterator.iterate(iteration_loops=1000001)

    # Play game with generated policies and save results!
    play_game_with_policies(env_name=ENV_NAME,
                            list_of_policies=list_of_policies,
                            output_file_path='outputs/policy_iter_GLIE_MC.png')

    final_Q = list_of_Qs[-1]
    plot_Q_func(Q_func=final_Q, output_path='outputs/optimal_policy_GLIE_MC')
