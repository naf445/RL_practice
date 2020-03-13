import sys
sys.path.append('..')
import numpy as np
import algorithms.dynamic_programming.policy_evaluation as pe

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
        List of [S, A] shaped matrices representing the policies found
            along the way to the final optimal policy found
    """

    def __init__(self, env, evaluation_loops, theta, discount_factor):
        # Create your policy evaluator you will use in prediction steps
        self.policy_evaluator = pe.policy_evaluator(env=env,
                                                    k_loops=evaluation_loops,
                                                    theta=theta,
                                                    discount_factor=discount_factor)
        self.nS = env.nS
        self.nA = env.nA
        self.env_dynamics = env.P # List of transition tuples (prob, next_state, reward, done)
        self.discount_factor = discount_factor

    def iterate(self, iteration_loops):
        """
        Main loop of policy iteration. Seeds an initial arbitrary policy.
        Then performs policy evaluation on this policy. Then greedily picks a new policy
        based on this updated value function. And policy evaluates this.
        "And so on to viscosity" - Lupe Fiasco
        """
        current_policy = np.zeros([self.nS, self.nA])  
        current_policy.fill(1/self.nA) # Begin with arbitrary random policy
        iterations_completed = 0
        policies_same = False
        list_of_policies = [current_policy.copy()] # Maintain policies throughout for comparison later
        while not policies_same and iterations_completed < iteration_loops:
            updated_value_function = self.policy_evaluator.evaluate(policy=current_policy) # Use policy evaluator to update value function
            new_policy = self.greedily_update_policy(updated_value_function=updated_value_function) # Use new value function to greedily get new policy
            policies_same = np.array_equal(new_policy, current_policy)
            current_policy = new_policy.copy()
            list_of_policies.append(current_policy.copy())
            iterations_completed += 1
        return list_of_policies 

    def greedily_update_policy(self, updated_value_function):
        """
        Given a value function, create a deterministic policy from it,
        simply by taking whichever action has the highest resulting Value function
        """
        q_values_all_states = [] # Holder for an self.nS x self.nA sized matrix which will be comprised of q-values
        for state in range(self.nS):
            q_values_indiv_state = [] # Will hold the q values for all actions for a given state
            for action in range(self.nA):
                action_value = 0
                state_actions_dynamics = self.env_dynamics[state][action]
                for state_action_tuple in state_actions_dynamics: # One of these for every possible state we could end up in
                    initial_reward = state_action_tuple[2]
                    next_state_prob = state_action_tuple[0]
                    next_state = state_action_tuple[1]
                    # Use the supplied value function to estimate the q values for all possible actions
                    state_contribution = initial_reward + self.discount_factor*(updated_value_function[next_state])
                    action_value += next_state_prob*state_contribution  
                q_values_indiv_state.append(action_value)
            q_values_all_states.append(q_values_indiv_state)
        q_matrix = np.array(q_values_all_states) # Turns our q values list of list in to a matrix for easier manipulation
        new_optimal_policy = np.eye(q_matrix.shape[-1])[np.argmax(q_matrix, axis=1)] # Just goes row by row and sets the max value to 1 and the other values to 0 to result in a deterministic policy!
        return new_optimal_policy 








