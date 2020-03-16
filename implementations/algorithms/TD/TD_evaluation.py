import numpy as np

class TD_lambda_backwards_eval(object):
    """
    Backward looking TD lambda prediction/evaluation algorithm.
    Calculates the value function for a given policy.

    Args:
        policy_function: A function that takes as inputs an agent's observation of environmental state and maps it to an action-probability tuple.
        env: OpenAI gym structured environment
        learning_rate: aka alpha; Controls the size of our value function update after we've determined the "direction" of our error
        lambda_value: Plays a role in eligibility function which determines structure of value function update algorithm
        num_episodes: Number of episodes to sample before stopping value function updating
        theta: Threshhold to stop value updating once our value function change is less than theta for all states
        discount_factor: Gamma discount factor

    Returns:
        A list with index corresponding to state and value corresponding to V(s)
    """

    def __init__(self, policy_function, env, learning_rate, lambda_value, num_episodes, theta, discount_factor):
        self.policy_function = policy_function
        self.env = env
        self.nS = env.env.nS # Number of states in the environment 
        self.nA = env.env.nA #  Number of actions in the environment
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.num_episodes = num_episodes
        self.theta = theta
        self.discount_factor = discount_factor

    def evaluate(self):
        """
        Main loop of the policy evaluation performed in backwards TD-lambda.
        Seeds an initial arbitrary value function, and then iteratively
        performs v(s) updates until stopping condition is met.
        """
        eligibility_function = np.zeros(self.nS) # Initial, 0 for eligibility for update for all states.
        # On each step, decay all state's eligibilities by γλ and increment the eligibility trace for the current state by 1
        v_func_current = np.zeros(self.nS) # Initial, arbitrary value_function for every state
        v_func_new = np.zeros(self.nS)
        delta_v_funcs = 1
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes and np.abs(np.max(delta_v_funcs)) > self.theta: # Loop breaks if either condition is violated
            state = self.env.reset()
            done = False
            states_visited = []
            while not done:
                states_visited.append(state)
                action = self.policy_function(state)
                next_state, reward, done, info = self.env.step(action)
                td_error = reward + self.lambda_value*v_func_new[next_state] - v_func_new[state] # Get TD_error term, indicates the step direction for us to update our value function estimate by
                eligibility_function[state] = eligibility_function[state]+1
                # Update value function at all states using current best estimate of value function and TD lambda formula which should mostly just adjust high eligibility states
                for state_visited in set(states_visited): # Only loop through previously visited states... backwards!
                    v_func_new[state_visited] =  v_func_new[state_visited] + self.alpha*eligibility_function[state_visited]*td_error
                    eligibility_function[state_visited] = eligibility_function[state_visited]*self.lambda_value*self.discount_factor
                state = next_state
            delta_v_funcs = v_func_new - v_func_current
            v_func_current = v_func_new.copy()
            episodes_sampled += 1
        return v_func_new
