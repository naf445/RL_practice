import numpy as np
from collections import defaultdict
import random

class TD_lambda_backwards_eval_SARSA(object):
    """
    Backward looking TD lambda prediction/evaluation algorithm.
    Calculates the action-value function for a given policy.

    Args:
        env: OpenAI gym structured environment
        learning_rate: aka alpha; Controls the size of our value function update after we've determined the "direction" of our error
        lambda_value: Plays a role in eligibility function which determines structure of value function update algorithm
        discount_factor: Gamma discount factor
    """

    def __init__(self, env, learning_rate, lambda_value, discount_factor):
        self.env = env
        self.nA = env.action_space.n
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.discount_factor = discount_factor

    def evaluate(self, policy_function, num_episodes, existing_Q=None, existing_E=None):
        """
        Main loop of the policy evaluation performed in backwards TD-lambda.
        Seeds an initial arbitrary action-value function, and then iteratively
        performs q(s) updates until stopping condition is met.

        Args:
            policy_function: A function that takes as inputs an agent's observation of environmental state and maps it to an action-probability tuple.
            num_episodes: Number of episodes to sample before stopping action-value function updating
            existing_Q: The existing Q function to further optimize.
            If None, initialize Q(s,a) to 0's.
            existing_E: The existing Eligibility Trace function to work with.
            If None, initialize E(s,a) to 0's.

        Returns:
            A dictionary that maps from action-state -> value.
        """

        if existing_E:
            E_func = existing_E
        else:
            E_func = defaultdict(lambda: [0.0]*self.nA) # Initial, 0 for eligibility for update for all states.
        # On each step, decay all state's eligibilities by γλ and increment the eligibility trace for the current state by 1
        if existing_Q:
            Q_func = existing_Q
        else:
            Q_func = defaultdict(lambda: [random.random() for num in range(self.nA)]) # New (s,a) gets default q function values
        episodes_sampled = 0
        while episodes_sampled < num_episodes: # Loop breaks if either condition is violated
            state = self.env.reset()
            done = False
            action_states_visited = []
            while not done:
                action = policy_function(state)
                action_states_visited.append((action, state))
                next_state, reward, done, info = self.env.step(action)
                next_action = policy_function(next_state)
                td_error = reward + self.discount_factor*Q_func[next_state][next_action] - Q_func[state][action] # Get TD_error term, indicates the step direction for us to update our value function estimate by
                E_func[state][action] += 1
                # Update q-function at all action-states using current best estimate of value function and TD lambda formula which should mostly just adjust high eligibility states
                for a, s in set(action_states_visited): # Only loop through previously visited states... backwards!
                    Q_func[s][a] = Q_func[s][a] + self.learning_rate*E_func[s][a]*td_error
                    E_func[s][a] = E_func[s][a]*self.lambda_value*self.discount_factor
                state = next_state
            episodes_sampled += 1
            if episodes_sampled % 1000 == 0:
                print('Finished episode: {}'.format(episodes_sampled))
        return E_func, Q_func 
