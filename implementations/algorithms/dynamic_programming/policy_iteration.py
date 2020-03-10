import sys
sys.append('..')
import numpy as np
import policy_evaluation as pe

class policy_iterator(object):
    """
    Given a full MDP specification, iteratively applies policy evaluation and greedy
    policy updating based on the resulting value function to find an optimal policy
         
     Args:
        env: OpenAI env
        k_loops: Number of iterations of synchronous Bellman Expectation Equation updates to the value function performed during evaluation steps
        theta: Threshhold to stop policy evaluation steps once our value function change is less than for all states
        discount_factor: Gamma discount factor for future rewards
                                                                                     
     Returns:
        [S, A] shaped matrix representing the optimal policy found 
    """

    def __init__(self, env, evaluation_loops, theta, discount_factor):
        self.policy_evaluator = pe.policy_evaluator(env=env,
                                                    k_loops=evaluation_loops,
                                                    theta=theta,
                                                    discount_factor=discount_factor)
        self.nS = env.nS
        self.nA = env.nA
        self.env_dynamics = env.P

    def iterate(iteration_loops):
        """
        Main loop of policy iteration. Seeds an initial arbitrary policy.
        Then performs policy evaluation on this policy. Then greedily picks a new policy
        based on this updated value function. And policy evaluates this.
        "And so on to infinity" - Lupe Fiasco
        """
        current_policy = np.zeros([nS, nA])  
        iterations_completed = 0
        policies_same = False
        while not policies_same and iterations_completed < iteration_loops:
            updated_value_function = self.policy_evaluator.evaluate(policy=current_policy)
            new_policy = greedily_update_policy(updated_value_function=updated_value_function)
            policies_same = np.array_equal(new_policy, current_policy)
            current_policy = new_policy

    def greedily_update_policy(self, updated_value_function):
        q_values_all_states = []
        for state in range(self.nS):
            q_values_indiv_state = []
            for action in range(self.nA):
                action_value = 0
                state_actions_dynamics = self.env_dynamics[state][action]
                for state_action_tuple in state_actions_dynamics:
                    initial_reward = state_action_tuple[2]
                    next_state_prob = state_action_tuple[0]
                    next_state = state_action_tuple[1]
                    state_contribution = initial_reward + self.discount_factor*(updated_value_function[next_state])
                    action_value += next_state_prob*state_contribution  
                q_values_indiv_state.append(action_value)
            q_values_all_states.append(q_values_indiv_state)
            q_matrix = np.array(q_values_all_states)
        return np.eye(q_matrix.shape[0])[np.argmax(q_matrix, axis=1)]
