import numpy as np

class MC_evaluator(object):
    """
    MC prediction/evaluation algorithm. Calculates the value function
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
        v_func_current = np.zeros(self.nS) # Initial, arbitrary value_function for every state
        v_func_new = np.ones(self.nS) 
        delta_v_funcs = v_func_new - v_func_current
        episodes_sampled = 0
        while episodes_sampled < self.num_episodes and np.max(delta_v_funcs) > self.theta: # Loop breaks if either condition is violated
            episode_states = [] 
            episode_actions = []
            episode_rewards = []
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy_function(state)
                episode_states.append(state)
                episode_actions.append(action)
                state, reward, done, info = lake_env.step(next_action) 
                episode_reward = reward
            # V update
            delta_v_funcs = v_func_new - v_func_current
            v_func_current = v_func_new















