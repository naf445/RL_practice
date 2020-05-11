import sys
sys.path.append('..')
import numpy as np
from collections import defaultdict
import numpy as np
import random
from copy import deepcopy
import algorithms.MC.MC_evaluation_online_Q as MCeval

class MC_on_policy(object):
    """
    This is an implementation of a MC policy iteration algorithm, using an on-policy strategy.
    The evaluation loop is going to be monte carlo evaluation of one episode at a time, updating Q(v,s) in an online fashion.
    The improvement step will follow an epsilon greedy strategy.
         
    Args:
        env: OpenAI env
        improvement_rounds: Number of rounds of policy evaluation-->improvement complete cycles before termination
        discount_factor: Gamma discount factor for future rewards
        eval_episodes: Number of episods for the evaluation loop to use to update the Q(s,a) function
                                                                                     
    Returns:
        List of tabular policy dictionaries, each of the form {(s1, s2, s3): action, (s4, s5, s6): action}
        along the way to the final policy found
    """

    def __init__(self, env, discount_factor, eval_episodes):
        # Create your policy evaluator you will use in prediction steps
        self.policy_evaluator = MCeval.MC_online_Q_evaluator(env=env,
                                                            num_episodes=eval_episodes,
                                                            discount_factor=discount_factor)
        self.nA = env.action_space.n

    def iterate(self, iteration_loops, print_every_n):
        """
        Main loop of this policy iteration.
        Performs policy evaluation to update an action-value function.
        Then epsilon-greedily picks a new policy based on this updated value function.
        Repeat a number of times.
        "And so on to viscosity" - Lupe Fiasco
        """
        current_policy_function = lambda state_input: random.randint(0,self.nA-1) # Initialize with random policy
        current_Q = None
        iterations_completed = 0
        list_of_Q_dicts = [] # Maintain Qs throughout for comparison later
        while iterations_completed < iteration_loops:
            current_Q = self.policy_evaluator.evaluate(policy_function=current_policy_function, existing_Q=current_Q) # Use policy evaluator to update action-value function
            current_policy_function = self.epsilon_greedily_update_policy(current_Q=current_Q,
                                                             iterations_completed=iterations_completed) # Use new Q function to epsilon greedily get new policy
            if iterations_completed % print_every_n == 0:
                print("Completed {} full policy iterations and saved most recent policy".format(iterations_completed))
                print(current_Q)
                list_of_Q_dicts.append(deepcopy(current_Q))
            iterations_completed += 1
        return list_of_Q_dicts

    def epsilon_greedily_update_policy(self, current_Q, iterations_completed):
        """
        Given a q function and number of loops completed,
        create an epsilon greedy policy from it.
        Returns a policy function which chooses an action from state.
        Epsilon = 1/iteration_number
        """
        iteration = iterations_completed+1
        # epsilon = min(1/np.log(iterations_completed+.0001),1)
        # epsilon = 1/iteration
        epsilon = 0.1
        def new_policy(state):
            heads = True if random.random() < epsilon else False # Flip our epsilon greedy coin
            if heads: # If heads comes up, choose random action
                return random.randint(0, self.nA-1)
            else: # If tails comes up, choose greedy option
                return np.argmax(current_Q[state]['Q(s,a)'])
        return new_policy



