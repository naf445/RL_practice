import numpy as np
from collections import defaultdict

class MC_online_Q_evaluator(object):
    """
    Every visit online MC prediction/eval algorithm.
    Calculates the action-value function for a given policy.

    Args:
        existing_Q: The existing Q function (and count of how many
            times we've been to the (s,a)'s) to further optimize.
            If None, initialize Q(S,A) to 0's.
        policy_function: A function that takes as inputs an agent's observation of environmental state and maps it to an action-probability tuple.
        env: OpenAI gym structured environment.
        num_episodes: Number of episodes to sample before stopping value function updating.
        discount_factor: aka Gamma/temporal discount factor.

    Returns:
        A dictionary that maps from action-state -> value.
    """

    def __init__(self, env, num_episodes, discount_factor):
        self.env = env
        self.nA = env.action_space.n
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor

    def create_new_Q(self):
        '''During a new action-state being encountered, we want to add this
        (a,s) to our tabular Q function. Also we want to keep track
        of the number of visits to to each of these. This function will be used
        in a defaultdict object to accomplish this aim.'''

        return {'Q(s,a)':[0.0]*self.nA,
                'num_visits':[0.0]*self.nA}

    def evaluate(self, policy_function, existing_Q=None):
        """
        Main loop of the policy evaluation performed via every visit online MC method.
        Updates the action value function after every episode in an online fashion.
        This online fashion means we don't need stores of past G's, only the current.
        Seeds an initial arbitrary action_value function,
        or continues from a supplied version and then iteratively
        performs Q(s) updates after every episode until stopping condition is met.
        """
        if existing_Q:
            Q_func = existing_Q
        else:
            Q_func = defaultdict(self.create_new_Q) # Initial, arbitrary value_function for every (s,a)
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes: # When loop breaks, policy is returned 
            state = self.env.reset()
            done = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            # Collect trajectory behaving with self.policy_function
            while not done:
                action = policy_function(state)
                episode_states.append(state)
                episode_actions.append(action)
                state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)
            for time_step in range(len(episode_states)): # Loop through every time step's actions/states/rewards and calculate G's
                # Get actual state & actions we will want update in this Q(s,a) 
                current_state = episode_states[time_step]
                current_action = episode_actions[time_step]
                current_G = episode_rewards[time_step:] # Get a list of immediate reward until end of trajectory for every state visit
                for R_index in range(len(current_G)): # Go through this list of rewards and apply discount factor in to the future
                    current_G[R_index] = current_G[R_index]*(self.discount_factor**R_index) # Apply future discounting before summing the R for a state visited
                current_G = np.sum(current_G)    
                # Q(s,a) update!
                Q_func[current_state]['num_visits'][current_action] += 1
                Q_func[current_state]['Q(s,a)'][current_action] = Q_func[current_state]['Q(s,a)'][current_action] + (1/Q_func[current_state]['num_visits'][current_action])*(current_G - Q_func[current_state]['Q(s,a)'][current_action])
            episodes_sampled += 1
        return Q_func 
