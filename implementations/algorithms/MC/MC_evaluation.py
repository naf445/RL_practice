import numpy as np

class MC_evaluator(object):
    """
    Every visit MC prediction/evaluation algorithm. Calculates the value function
    for a given policy.

    Args:
        policy_function: A function that takes as inputs an agent's observation of environmental state and maps it to an action-probability tuple.
        env: OpenAI gym structured environment
        learning_rate: aka alpha; Controls the size of our value function update after error direction has been determined
        num_episodes: Number of episodes to sample before stopping value function updating
        theta: Threshhold to stop value updating once our value function change is less than theta for all states
        discount_factor: Gamma/temporal discount factor

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    def __init__(self, policy_function, env, learning_rate, num_episodes, theta, discount_factor):
        self.policy_function = policy_function
        self.env = env
        self.nS = env.env.nS # Number of states in the environment 
        self.nA = env.env.nA #  Number of actions in the environment
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.theta = theta
        self.discount_factor = discount_factor

    def evaluate(self):
        """
        Main loop of the policy evaluation performed via MC method.
        Seeds an initial arbitrary value function, and then iteratively
        performs synchronous v(s) updates after every episode until stopping condition is met.
        """
        v_func_current_average = np.zeros(self.nS) # Initial, arbitrary value_function for every state
        # Because states can be visited multiple times in an episode and this is an every state monte carlo, we need lists to capture R's for every visit to a particular state
        v_func_new_raw_values = [[] for num in range(self.nS)]
        v_func_new_average = np.zeros(self.nS)
        delta_v_funcs = 1 
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes and np.max(delta_v_funcs) > self.theta: # Loop breaks if either condition is violated
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
                state, reward, done, info = env.step(next_action)
                episode_rewards.append(reward)
            for episode_states_index in range(len(episode_states)): # Loop through every state we found ourselves in
                current_state = episode_states[episode_states_index] # Get actual state value, we will want to append our R in this V(state) 
                R_from_state_visit = episode_rewards[index:] # Get a list of immediate reward until end of trajectory for every state visit
                for R_index in range(len(R_from_state_visit)): # Go through this list of rewards and apply discount factor in to the future
                    R_from_state_visit[R_index] = R_from_state_visit[R_index]*(self.discount_factor^R_index) 
                v_func_new[current_state].append(np.sum(R_from_state_visit)) # For every state we found ourselves in, append the R from that visit to the value function list
            v_func_new = [np.ave(reward_list) for reward_list in v_func_new] # Update value function by 
            delta_v_funcs = v_func_new - v_func_current
            v_func_current = v_func_new
            episodes_samples += 1

