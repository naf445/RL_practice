import numpy as np
from collections import defaultdict
import random

class eval_SARSA_0(object):
    """
    TD 0 prediction/evaluation algorithm.
    Calculates the action-value function for a given policy.

    Args:
        env: OpenAI gym structured environment
        learning_rate: aka alpha; Controls the size of our value function update after we've determined the "direction" of our error
        lambda_value: Plays a role in eligibility function which determines structure of value function update algorithm
        discount_factor: Gamma discount factor
    """

    def __init__(self, env, learning_rate, discount_factor):
        self.env = env
        self.nA = env.action_space.n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    
    def policy_function(self, state, current_Q, epsilon_soft):
        """
        Given a q function, creates an epsilon policy from it.
        """
        epsilon = epsilon_soft # Epsilon soft policy, always this epsilon random chance
        heads = True if random.random() < epsilon else False # Flip our epsilon greedy coin
        if heads: # If heads comes up, choose random action
            return random.randint(0, self.nA-1)
        else: # If tails comes up, choose greedy option
            return np.argmax(current_Q[state])

    def evaluate(self, num_episodes, epsilon_soft, existing_Q=None):
        """
        Main loop of the policy evaluation performed in backwards TD-0.
        Seeds an initial arbitrary action-value function, and then iteratively
        performs q(s) updates until stopping condition is met.

        Args:
            num_episodes: Number of episodes to sample before stopping action-value function updating
            existing_Q: The existing Q function to further optimize.
            If None, initialize Q(s,a) to 0's.

        Returns:
            A dictionary that maps from action-state -> value.
        """

        if existing_Q:
            Q_func = existing_Q
        else:
            Q_func = defaultdict(lambda: [random.random() for num in range(self.nA)]) # New (s,a) gets default q function values
        episodes_sampled = 0
        while episodes_sampled < num_episodes: # Loop breaks if either condition is violated
            state = self.env.reset() # S in sarsa
            done = False
            action_states_visited = []
            while not done:
                action = self.policy_function(state=state,
                                              current_Q=Q_func,
                                              epsilon_soft=epsilon_soft) # A in sarsa
                next_state, reward, done, info = self.env.step(action) # R and S' in sarsa
                next_action = self.policy_function(state=next_state,
                                                   current_Q=Q_func,
                                                   epsilon_soft=epsilon_soft) # A' in sarsa
                td_error = reward + self.discount_factor*Q_func[next_state][next_action] - Q_func[state][action] # Get TD_error term, indicates the step direction for us to update our value function estimate by
                Q_func[state][action] = Q_func[state][action] + self.learning_rate*td_error
                state = next_state # S in sarsa
            episodes_sampled += 1
            if episodes_sampled % 1000 == 0:
                print('Finished episode: {}'.format(episodes_sampled))
        return Q_func 
