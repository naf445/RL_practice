import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)

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
        logging.info('created linear function approximator object')
        logging.info('initial weights:\n{}'.format(self.weights))
    
    def predict(self, features_array, action_array=None):
        """
        This function will run features through the current linear function.
        If no action_array is provided, it will return an array of size num_actions,
        with each spot being the current q_value estimate for that action.
        If an action_array is provided, it is assumed you want an array of the q_values
        for the specific action specified.

        action_array : numpy array of dimension num_steps x num_actions 
        """
        if action_array is not None:
            logging.debug('action_array provided')
            logging.debug('self.weights: {}'.format(self.weights))
            logging.debug('features_array: {}'.format(features_array))
            logging.debug('action_array: {}'.format(action_array))
            return np.multiply(action_array.dot(self.weights), features_array).sum(axis=1)
        else:
            logging.debug('no action_array provided!!')
            logging.debug('self.weights: {}'.format(self.weights))
            logging.debug('features_array: {}'.format(features_array))
            return np.dot(self.weights, features_array)

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
        logging.debug('Beginning weight optimization!')
        logging.debug('features_array: {}'.format(features_array))
        logging.debug('action_array: {}'.format(action_array))
        logging.debug('weights: {}'.format(self.weights))
        estimated_action_values = self.predict(features_array=features_array, action_array=action_array)
        logging.debug('estimated_action_values: {}'.format(estimated_action_values))
        logging.debug('true_action_values: {}'.format(true_action_values))
        error_vector = true_action_values - estimated_action_values
        logging.debug('error vector: {}'.format(error_vector))
        # We should have a tensor of depth num_steps with each matrix being [num_actions, num_features]
        gradient = np.einsum('ij, ik-> ikj', features_array, action_array) # This is a little confusing, but basically we want to do row by row multiplication of our feature and action arrays and store the resulting matrices in a tensor
        logging.debug('current gradient form: {}'.format(gradient))
        gradient = np.einsum('i, ikj-> ikj', error_vector, gradient) # Here we want to multiply all of our matrices a scalar, which is that step's error value!
        logging.debug('current gradient form: {}'.format(gradient))
        gradient = gradient*(-2) 
        logging.debug('current gradient form: {}'.format(gradient))
        gradient = np.sum(gradient, axis=0)
        logging.debug('current gradient form: {}'.format(gradient))

        # Step 6) Pick & Apply an update rule
        """ Ok, now we have a weight shaped matrix called gradient, which tells us the dLoss/dWeight for all of our weights.
        We will now shift our initial weight guesses by a certain amount. If our gradient is positive, that means our weight guess
        is currently too large, and we need to lower it, and vice versa. Thus we flip around the sign of the items currently
        in our gradient object. Also let's multiply by a learning rate to avoid large steps and reduce noise! 
        """
        delta_matrix = gradient*(-1)
        logging.debug('current delta_matrix: {}'.format(delta_matrix))
        delta_matrix *= learning_rate
        logging.debug('current delta_matrix: {}'.format(delta_matrix))
        logging.debug('current weights: {}'.format(self.weights))
        self.weights += delta_matrix
        logging.debug('weights after update: {}'.format(self.weights))
        
