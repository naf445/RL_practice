import numpy as np
import random

class linear_function_approximator(object):
    """
    This is an implementation of linear function approximation.
    This class will hold the actual linear function, as well
    as perform optimization of parameters via gradient descent.
    """
    
    def __init__(self, num_actions, num_features):
        self.num_features = num_features
        self.num_actions = num_actions 
        self.weights = np.random.rand(self.num_actions, self.num_features) # Because this model will predict the action values for ALL actions every time, we will need a parameter for every feature for every possible action!
    
    def predict(self, features_array, action_array):
        """
        action_array : numpy array of dimension num_episodes x num_actions 
        """
        return np.multiply(action_array.dot(weights), features_array).sum(axis=1)

    def update_weights(self, features_array, action_array, true_action_values, learning_rate):
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

        # Step 4) Set initial parameter value guesses. This is our self.weights, we can assume this is done
        # Step 5) Get slope values for all parameters 
        estimated_action_values = self.predict(features_array=features_array, action_array=action_array)
        error_vector = true_action_values - estimated_action_values
        # We should have a tensor of depth num_episodes with each matrix being [num_actions, num_features]
        gradient = np.einsum('ij, ik-> ikj', features_array, action_array) # This is a little confusing, but basically we want to do row by row multiplication of our feature and action arrays and store the resulting matrices in a tensor
        gradient = np.einsum('i, ikj-> ikj', error_vector, gradient) # Here we want to multiply all of our matrices a scalar, which is that episode's error value!
        gradient = gradient*(-2) 
        gradient = np.sum(gradient, axis=0)

        # Step 6) Pick & Apply an update rule
        """ Ok, now we have a weight shaped matrix called gradient, which tells us the dLoss/dWeight for all of our weights.
        We will now shift our initial weight guesses by a certain amount. If our gradient is positive, that means our weight guess
        is currently too large, and we need to lower it, and vice versa. Thus we flip around the sign of the items currently
        in our gradient object. Also let's multiply by a learning rate to avoid large steps and reduce noise! 
        """
        delta_matrix = gradient*(-1)
        delta_matrix *= learning_rate
        self.weights += delta_matrix
        



