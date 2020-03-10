import numpy as np

class value_iterator(object):
    """
        Given a full MDP specification, iteratively applies updates on a value function 
        according to the Bellman Optimality Equation

        Args:
            env: OpenAI env
            k_loops: Number of iterations of synchronous Bellman Optimality Equation
                updates to be performed
            theta: Threshhold to stop once our value function
                change is less than theta for all states
            discount_factor: Gamma discount factor for future rewards

        Returns:
            [S, A] shaped matrix representing the optimal policy found  
    """
 
    def value_iterate(self, env, k_loops, theta, discount_factor):
        self.env_dynamics = env.P # List of transition tuples (prob, next_state, reward, done)
        self.nS = env.nS # Number of states in the environment
        self.nA = env.nA #  Number of actions in the environment
        self.k_loops = k_loops # Number of loops to continue evaluation for
        self.theta = theta # Cutoff for saying value function has not changed since the past version prior to most recent iteration
        self.discount_factor = discount_factor 
        
        v_star_current = np.zeros(self.nS) # Initial, arbitrary v_star for all states
        v_star_new = np.ones(self.nS)
        delta_v_stars = v_star_new - v_star_current
        optimal_policies = []
        iterations_complete = 0
        while iterations_complete < self.k_loops and np.max(delta_v_stars) > self.theta:
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
                        state_contribution = initial_reward + self.discount_factor*(v_star_current[next_state])
                        action_value += next_state_prob*state_contribution  
                    q_values_indiv_state.append(action_value)
                q_values_all_states.append(q_values_indiv_state)
            q_matrix = np.array(q_values_all_states) # Turns our q values list of list in to a matrix for easier manipulation
            new_optimal_policy = np.eye(q_matrix.shape[-1])[np.argmax(q_matrix, axis=1)] # Just goes row by row and sets the max value to 1 and the other values to 0 to result in a deterministic policy!
            optimal_policies.append(new_optimal_policy)
            v_star_new = np.max(q_matrix, axis=1) 
            delta_v_stars = v_star_new - v_star_current 
            v_star_current = v_star_new.copy() 
            iterations_complete+=1
        return optimal_policies

