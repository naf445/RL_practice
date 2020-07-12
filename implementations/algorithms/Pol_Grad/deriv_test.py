import numpy as np


class softmax_policy(object):

    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.weights = np.random.rand(self.nS, self.nA)

    def softmax(self, input_array):
        exp_inputs = np.exp(input_array)
        sum_exp_inputs = np.sum(exp_inputs)
        return exp_inputs/sum_exp_inputs

    def get_action_probs(self, obs):
        return softmax(obs.dot(self.weights))

    def softmax_grad(action_array):
        # For the softmax derivative/gradient, we don't just get one equation
        # that contains this thing. Because there are multiple inputs and multiple outputs
        # we actually will be getting the Jacobian. This Jacobian contains information which
        # is the derivative of an output w.r.t. one of the inputs, and it covers
        # each individual outputs w.r.t. each individual inputs. This leads to
        # a Jacobian matrix of dimension [n_inputs x n_outputs]
        # For our notation, we will refer to an individual input as a_i and 
        # an output as s_j. 
        # We can refer to the opex daily handbook for the specific, but on the diagonal
        # of this matrix, we will have cases where we are taking the derivative of
        # an s_i and an a_j where i=j, which I mention because there are 2 derivative
        # equations. One where i=j and whon where i!=j.
        # Again these specific equations for deriv of s_j w.r.t. a_i can be found in opex daily.
        jacobian = np.zeros([self.nA, self.nA])
        for row_index in range(self.nA):
            for column_index in range(self.nA):
                if row_index==column_index: # You are in an i=j case
                    s_i =  action_array[row_index]
                    jacobian[row_index, column_index] = s_i*(1-s_i)
                else:
                    s_i =  action_array[row_index]
                    s_j =  action_array[column_index]
                    jacobian[row_index, column_index] = -1*s_i*s_j
        return jacobian


    def update_weights(self, state_obs, action_array, value):

        grad_mat_mult = state_obs
        grad_pol = grad_softmax*grad_mat_mult 
        grad_log_pol=(1/action_array)*grad_pol
        delta_matrix = learning_rate*grad_log_pol*value
        self.weights += delta_matrix
        




