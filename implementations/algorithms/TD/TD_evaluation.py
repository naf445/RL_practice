import numpy as np

class TD_predictor(policy_function, env, learning_rate, forward_mode, lambda_value, num_episodes, discount_factor):
    """
    Flexible TD lambda prediction/evaluation algorithm. Calculates the value function
    for a given policy. With this object, it is possible to tune the settings
    to use one of 5 possible algorithms. The algorithms allowed and settings
    to achieve them are detailed below:
        1) Monte Carlo
            - forward_mode=True, lambda_value=1 
        2) TD(0)
            - lambda_value=0
        3) TD(1)  
            - forward_mode=False, lambda_value=1
        4) Forward TD(lambda)
            - forward_mode=True, lambda_value=(0,1)
        5) Backward TD(lambda) 
            - forward_mode=False, lambda_value=(0,1)

    Args:
        policy_function: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        learning_rate: aka alpha;
        forward_mode:
        lambda_value:
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
