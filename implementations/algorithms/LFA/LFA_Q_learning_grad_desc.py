import numpy as np
import random

class linear_function_approximator(object):
    """
    This is an implementation of linear function approximation.
    This class will hold the actual linear function, as well
    as perform optimization of parameters via gradient descent.
    """
    
    def __init__(self, num_actions, num_features, learning_rate):
        self.num_features = num_features
        self.num_actions = num_actions 
        self.weights = np.random.rand(self.num_features, self.num_actions)
        self.learning_rate = learning_rate
    
    def predict(self, features, action):
        """
        action : numpy array of the length of num_actions with a 1 in the chosen action
        """
        return self.weights.dot(action).dot(features)

    def optimize(self, features, action, true_action_value):
        """ This is the meat and potatoes, where you will implement the gradient descent.
        Let's go through the steps of gradient descent as covered in your opex daily
        book on page 4-16.
        """
        # Step 1) Define our big picture function
        """ 1) 
        estimated_action_value = weight_1_1*feature_1*action_1 + weight_2_1*feature_2*action_1 + ... weight_n_1*feature_n*action_1 + 
                       weight_1_2*feature_1*action_2 + weight_2_2*feature_2*action_2 + ... weight_n_2*feature_n*action_2 + 
                       weight_1_n*feature_1*action_n + weight_2_n*feature_2*action_n + ... weight_n_n*feature_n*action_n 
        """
        
        # Step 2) Pick our Loss Function
        """ 2)
        Loss Function = SSE = Sum[(true_action_value - estimated_action_value)^2]
        estimated_action_value is found using equation (1)
        """
        
        # Step 3) For all parameters, calculate derivative w.r.t. Loss Function from (2). See Opex Daily Book, entry 5-11 for handwritten details
        """ We have num_features * num_actions parameters/weights to estimate/tune/optimize 
        Example, weight_n_1:
          --> dLoss/dweight_n_1 = d/dweight_n_1(Sum[(true_action_value - estimated_action_value)^2]) 
          --> Sum[-2(true_action_value - estimated_action_value)(feature_n*action_1)]
        """

        # Step 4) 


