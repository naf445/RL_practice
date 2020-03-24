import sys
sys.path.append('..')
import numpy as np
from collections import defaultdict
import numpy as np
import random
from copy import deepcopy
import algorithms.TD.TD_lambda_backwards_eval_Q_learning as TDeval

class TD_off_policy(object):
    """
    This is an implementation of a TD policy iteration algorithm, using an off-policy strategy. AKA Q-learning
    The evaluation loop is going to be backwards TD lambda evaluation of one episode at a time, updating Q(v,s) in an online fashion.
         
    Args:
        env: OpenAI env
        improvement_rounds: Number of rounds of policy evaluation-->improvement complete cycles before termination
        discount_factor: Gamma discount factor for future rewards
        learning_rate: During evaluation step, aka alpha, controls size of our q function update
        lambda_value: During evaluation step, plays a role in eligibility decay function
        eval_episodes: Number of episods for the evaluation loop to use to update the Q(s,a) function
                                                                                     
    Returns:
        List of tabular policy dictionaries, each of the form {(s1, s2, s3): action, (s4, s5, s6): action}
        along the way to the final policy found
    """

    def __init__(self, env, discount_factor, learning_rate, lambda_value):
        # Create your policy evaluator you will use in prediction steps
        self.policy_evaluator = TDeval.TD_lambda_backwards_eval_Q_learning(env=env,
                                                    learning_rate=learning_rate,
                                                    lambda_value=lambda_value,
                                                    discount_factor=discount_factor)
        self.nA = env.action_space.n
        print('''---Creating Iterator Object---
discount factor: {}
learning rate (alpha): {}
lambda: {}'''.format(discount_factor, learning_rate, lambda_value))

    def iterate(self, iteration_loops, print_every_n, eval_episodes, epsilon_soft):
        """
        Main loop of this policy iteration.
        Performs policy evaluation to update an action-value function.
        """
        current_Q = None
        current_E = None
        iterations_completed = 0
        list_of_Q_dicts = [] # Maintain Qs throughout for comparison later
        while iterations_completed < iteration_loops:
            current_E, current_Q = self.policy_evaluator.evaluate(num_episodes=eval_episodes, epsilon_soft=epsilon_soft, existing_Q=current_Q, existing_E=current_E) # Use policy evaluator to update action-value function
            if iterations_completed % print_every_n == 0:
                print("Completed {} full policy iterations and saved most recent policy".format(iterations_completed))
                print(current_Q)
                list_of_Q_dicts.append(deepcopy(current_Q))
            iterations_completed += 1
        return list_of_Q_dicts




