import numpy as np
from collections import defaultdict

class MC_evaluator(object):
    """
    Every visit MC prediction/evaluation algorithm. Calculates the value function
    for a given policy.

    Args:
        policy_function: A function that takes as inputs an agent's observation of environmental state and maps it to an action-probability tuple.
        env: OpenAI gym structured environment
        num_episodes: Number of episodes to sample before stopping value function updating
        discount_factor: Gamma/temporal discount factor

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    def __init__(self, policy_function, env, learning_rate, num_episodes, discount_factor):
        self.policy_function = policy_function
        self.env = env
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor

    def evaluate(self):
        """
        Main loop of the policy evaluation performed via every visit MC method.
        Seeds an initial arbitrary value function, and then iteratively
        performs synchronous v(s) updates after every episode until stopping condition is met.
        """
        v_func_current_average = defaultdict(float) # Initial, arbitrary value_function for every state
        # Because states can be visited multiple times in an episode and this is an every state monte carlo, we need lists to capture R's for every visit to a particular state
        v_func_new_raw_values = defaultdict(list) 
        v_func_new_average = defaultdict(float)
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes: # Loop breaks if either condition is violated
            episode_states = []
            episode_actions = []
            episode_rewards = []
            state = self.env.reset()
            done = False
            # Collect trajectory, including states/actions/rewards at every time step
            while not done:
                action = self.policy_function(state)
                episode_states.append(state)
                episode_actions.append(action)
                state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)
            for episode_states_index in range(len(episode_states)): # Loop through indexes of list of every state we found ourselves in
                current_state = episode_states[episode_states_index] # Get actual state value, we will want to append our R in this V(state) 
                R_from_state_visit = episode_rewards[episode_states_index:] # Get a list of immediate reward until end of trajectory for every state visit
                for R_index in range(len(R_from_state_visit)): # Go through this list of rewards and apply discount factor in to the future
                    R_from_state_visit[R_index] = R_from_state_visit[R_index]*(self.discount_factor**R_index) # Apply future discounting before summing the R for a state visited
                v_func_new_raw_values[current_state].append(np.sum(R_from_state_visit)) # For every state we found ourselves in, append the R (including future discounting) from that visit to the value function list
            for key, value in v_func_new_raw_values.items():
                v_func_new_average[key] = np.average(value) # Update value function by averaging all reward streams found from a given state visit
            v_func_current_average = v_func_new_average.copy()
            episodes_sampled += 1
            if episodes_sampled % 10000 == 0:
                print('Finished episode: {}'.format(episodes_sampled))
        return v_func_new_average








