import numpy as np
from collections import defaultdict

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
        discount_factor: Gamma discount factor
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    def __init__(self, policy_function, env, learning_rate, lambda_value, num_episodes, discount_factor):
        self.policy_function = policy_function
        self.env = env
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor

    def evaluate(self):
        """
        Main loop of the policy evaluation performed in backwards TD-lambda.
        Seeds an initial arbitrary value function, and then iteratively
        performs v(s) updates until stopping condition is met.
        """
        eligibility_function = defaultdict(float) # Initial, 0 for eligibility for update for all states.
        # On each step, decay all state's eligibilities by γλ and increment the eligibility trace for the current state by 1
        v_func_new = defaultdict(float) 
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes: # Loop breaks if either condition is violated
            state = self.env.reset()
            done = False
            states_visited = []
            while not done:
                states_visited.append(state)
                action = self.policy_function(state)
                next_state, reward, done, info = self.env.step(action)
                td_error = reward + self.discount_factor*v_func_new[next_state] - v_func_new[state] # Get TD_error term, indicates the step direction for us to update our value function estimate by
                eligibility_function[state] = eligibility_function[state]+1
                # Update value function at all states using current best estimate of value function and TD lambda formula which should mostly just adjust high eligibility states
                for state_visited in set(states_visited): # Only loop through previously visited states... backwards!
                    v_func_new[state_visited] =  v_func_new[state_visited] + self.learning_rate*eligibility_function[state_visited]*td_error
                    eligibility_function[state_visited] = eligibility_function[state_visited]*self.lambda_value*self.discount_factor
                state = next_state
            episodes_sampled += 1
            if episodes_sampled % 50000 == 0:
                print('Finished episode: {}'.format(episodes_sampled))
        return v_func_new
