import sys
sys.path.append('..')
import numpy as np
from collections import defaultdict
import numpy as np
import random
from copy import deepcopy
import algorithms.TD.TD_lambda_backwards_eval_SARSA_0 as TDeval

class SARSA_0_on_policy(object):
    """
    This is an implementation of a TD policy iteration algorithm, using an on-policy strategy.
    The evaluation loop is going to be TD-0 evaluation of one episode at a time, updating Q(v,s) in an online fashion.
         
    Args:
        env: OpenAI env
        learning_rate: During evaluation step, aka alpha, controls size of our q function update
        discount_factor: Gamma discount factor for future rewards
        lambda_value: During evaluarion step, plays a role in eligibility decay function
                                                                                     
    Returns:
        List of tabular policy dictionaries, each of the form {(s1, s2, s3): action, (s4, s5, s6): action}
        along the way to the final Q_func found
    """

    def __init__(self, env, learning_rate, discount_factor):
        # Create your policy evaluator you will use in prediction steps
        self.policy_evaluator = TDeval.eval_SARSA_0(env=env,
                                                    learning_rate=learning_rate,
                                                    discount_factor=discount_factor)
        self.nA = env.action_space.n
        print('''---Creating Iterator Object---
discount factor: {}
learning rate (alpha): {}'''.format(discount_factor, learning_rate))

    def iterate(self, iteration_loops, print_every_n, eval_episodes, epsilon_soft):
        """
        Main loop of this policy iteration.
        Performs policy evaluation to update an action-value function.
        Repeat a number of times.
        "And so on to viscosity" - Lupe Fiasco
        """
        current_policy_function = lambda state_input: random.randint(0,self.nA-1) # Initialize with random policy
        current_Q = None
        iterations_completed = 0
        list_of_Q_dicts = [] # Maintain Qs throughout for comparison later
        while iterations_completed < iteration_loops:
            current_Q = self.policy_evaluator.evaluate(existing_Q=current_Q, num_episodes=eval_episodes, epsilon_soft=epsilon_soft) # Use policy evaluator to update action-value function
            if iterations_completed % print_every_n == 0:
                print("Completed {} full policy iterations and saved most recent policy".format(iterations_completed))
                print(current_Q)
                list_of_Q_dicts.append(deepcopy(current_Q))
            iterations_completed += 1
        return list_of_Q_dicts




