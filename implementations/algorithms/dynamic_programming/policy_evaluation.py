import numpy as np 

class policy_evaluator(object):
    """
    Given a full MDP specification and a policy to evaluate, give the value function at every state
    in the environment after k iterations updating the Bellman Expectation Equation.
        
    Args:
        policy: [S, A] shaped matrix representing the policy to evaluate
        env: OpenAI env
        k_loops: Number of iterations of synchronous Bellman Expectation Equation updates to the value function
        theta: Threshhold to stop evaluation once our value function change is less than for all states
        discount_factor: Gamma discount factor for future rewards
                                                                                    
    Returns:
        Vector of length env.nS representing the value function.
    """

    def __init__(self, policy, env, k_loops, theta, discount_factor):
        self.policy = policy
        self.env_dynamics = env.P # List of transition tuples (prob, next_state, reward, done)
        self.nS = env.nS # Number of states in the environment 
        self.nA = env.nA #  Number of actions in the environment
        self.k_loops = k_loops
        self.theta = theta
        self.discount_factor = discount_factor
        
    def evaluate(self):
        """
        Main loop of policy evaluation. Seeds an initial arbitrary value function, and then iteratively
        performs synchronous Bellman Expectation Equation updates until stopping condition is met.
        """
        v_func_current = np.zeros(self.nS) # Initial, arbitrary value_function for every state
        v_func_new = np.ones(self.nS) 
        delta_v_func = v_func_new - v_func_current
        iterations_complete = 0
        while iterations_complete < self.k_loops and np.max(delta_v_func) > self.theta:
            for state in range(self.nS): # Loop through all the states
                new_value_function_output = 0
                for action in range(self.nA):
                    action_probability = self.policy[state, action]
                    state_actions_dynamics = self.env_dynamics[state][action]
                    for state_action_tuple in state_actions_dynamics: # Transition tuples (prob, next_state, reward, done)
                        initial_reward = state_action_tuple[2]
                        next_state_prob = state_action_tuple[0]
                        next_state = state_action_tuple[1]
                        state_contribution = initial_reward + self.discount_factor*(v_func_current[next_state]) 
                        new_value_function_output += action_probability*next_state_prob*state_contribution
                v_func_new[state] = new_value_function_output
                delta_v_func = v_func_new - v_func_current
                iterations_complete += 1
        return v_func_new





