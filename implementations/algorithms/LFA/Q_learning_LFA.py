import sys
sys.path.append('..')
import numpy as np
import random
import algorithms.LFA.LFA as LFA
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)

class Q_learner(object):

    def __init__(self, env):
        self.env = env
        self.env.seed(244)
        self.nA = self.env.action_space.n
        self.nF = len(env.reset()) # Number of features in an observation
        self.LFA = LFA.linear_function_approximator(
                num_actions=self.nA,
                num_features=self.nF)
        logging.info('created LFA object')

    def iterate(self, training_episodes, steps_per_update, learning_rate, discount_factor, epsilon, print_every_n_episodes):
        LFA_list = []
        for episode in range(training_episodes):
            logging.info('beginning episode {}'.format(episode))
            # Get starting state
            state = self.env.reset()
            done = False
            features_array = []
            action_array = []
            true_action_values = []
            estimated_action_values = []
            step = 0
            while not done:
                logging.debug('step: {}'.format(step))
                features_array.append(state)
                # Get q values of all possible actions given current state
                estimated_q_values = self.LFA.predict(features_array=state)
                logging.debug('estimated_q_values: {}'.format(estimated_q_values))
                # Get index of 'best' action
                greedy_action = np.argmax(estimated_q_values)
                logging.debug('greedy_action: {}'.format(greedy_action))
                # Get index of epsilon action
                epsilon_action = greedy_action if random.random() > epsilon else random.randint(0, self.nA-1)
                logging.debug('epsilon_action: {}'.format(epsilon_action))
                # Create array from epsilon action of length nA with a 1 in the index
                epsilon_action_array = np.zeros(self.nA) 
                epsilon_action_array[epsilon_action] = 1
                # Add this array to list of ations
                action_array.append(epsilon_action_array)
                # Get Q-value of chosen epsilon action and save this q-value
                estimated_action_value = estimated_q_values[epsilon_action]
                estimated_action_values.append(estimated_action_value)
                # Execute action, get environment response 
                next_state, reward, done, info = self.env.step(epsilon_action) 
                # Get q values for all possible actions given 'next' state 
                next_q_values = self.LFA.predict(features_array=next_state)
                # Calculate the 'true' action value from immediate reward + bootstrapped best next action value
                true_action_value = reward+discount_factor*(np.max(next_q_values)) 
                true_action_values.append(true_action_value)
                state = next_state
                if episode>0 and print_every_n_episodes % episode == 0:
                    logging.info('episode {}'.format(episode))
                    logging.info('step: {}'.format(step))
                    logging.info('estimated_q_values: {}'.format(estimated_q_values))
                    logging.info('greedy_action: {}'.format(greedy_action))
                    logging.info('epsilon action: {}'.format(epsilon_action))
                    logging.info('epsilon action array: {}'.format(epsilon_action_array))
                    logging.info('greedy_action: {}'.format(greedy_action))
                    logging.info('estimated_action_value: {}'.format(estimated_action_value))
                    logging.info('true_action_value: {}'.format(true_action_value))
                    logging.info('features_array: {}'.format(features_array))
                    logging.info('action_array: {}'.format(action_array))
                    logging.info('true_action_values: {}'.format(true_action_values))
                    logging.info('estimated_action_values: {}'.format(estimated_action_values))

                if (step >0 and step % steps_per_update == 0) or done:
                    logging.debug('updating weights')
                    self.LFA.update_weights(features_array=np.array(features_array),
                            action_array=np.array(action_array),
                            true_action_values=np.array(true_action_values),
                            learning_rate=np.array(learning_rate)
                            )
                    features_array = []
                    action_array = []
                    true_action_values = []
                    estimated_action_values = []
                step+=1
            if episode % print_every_n_episodes==0 or episode==training_episodes-1:
                logging.info('appending episode: {}'.format(episode))
                LFA_list.append(deepcopy(self.LFA))
        return LFA_list
            

        




