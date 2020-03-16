import sys
sys.path.append('../../') 
import algorithms.MC.MC_evaluation as MC
import algorithms.TD.TD_lambda_backwards_evaluation as TD
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'Blackjack-v0'
MC_EVAL = False 
TD_EVAL = True

def MC_eval(env_name, policy_function, learning_rate, num_episodes, discount_factor):
    """The observation is a 3-tuple of: the players current sum,
    the dealer's up card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1)."""
    env = gym.make(env_name)
    env.seed(21)
    MC_evaluator_instance = MC.MC_evaluator(policy_function=policy_function, env=env, learning_rate=learning_rate, num_episodes=num_episodes, discount_factor=discount_factor) 
    MC_value_func = MC_evaluator_instance.evaluate()    
    return MC_value_func

def TD_eval(env_name, policy_function, learning_rate, lambda_value, num_episodes, discount_factor):
    env = gym.make(env_name)
    env.seed(21)
    TD_evaluator_instance = TD.TD_lambda_backwards_eval(policy_function=policy_function, env=env, learning_rate=learning_rate, lambda_value=lambda_value, num_episodes=num_episodes, discount_factor=discount_factor) 
    TD_value_func = TD_evaluator_instance.evaluate()    
    return TD_value_func

def plot_value_function(V, title, output_file):
    """
    Plots the value function as a surface plot.
    Code taken from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plt.savefig(fname=output_file+'_NoAce'+'.png')
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
    plt.savefig(fname=output_file+'Ace'+'.png')

def sample_policy_function(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

if __name__ == '__main__':

    if MC_EVAL:
        print('Running MC_eval')
        MC_value_func = MC_eval(env_name=ENV_NAME,
                                policy_function=sample_policy_function,
                                learning_rate=0.5,
                                num_episodes=50000,
                                discount_factor=0.95)

        plot_value_function(V=MC_value_func, title="MC Evaluation", output_file='outputs/MC_eval')

    if TD_EVAL:
        print('Running TD_eval')
        TD_value_func = TD_eval(env_name=ENV_NAME,
                                policy_function=sample_policy_function,
                                learning_rate=0.7,
                                lambda_value=0.7,
                                num_episodes=10000000,
                                discount_factor=0.3)

        plot_value_function(V=TD_value_func, title='TD Evaluation', output_file='outputs/TD_eval')








