import sys
sys.path.append('../../') 
import algorithms.dynamic_programming.policy_evaluation as pe 
import gym
import matplotlib.pyplot as plt
import numpy as np

lake_env = gym.make('FrozenLake8x8-v0')
lake_env.reset()
lake_env.seed(42)

policy_evaluator = pe.policy_evaluator(env=lake_env.env,
                                       k_loops=100,
                                       theta=0.001,
                                       discount_factor=0.95)


