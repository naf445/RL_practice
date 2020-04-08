# RL Course by David Silver, Lecture 6
## Notes by Nathan Franklin
## March-2020

- Wheeew we are finally back to taking some notes.
You have been on a hiatus implementing a lot of the algorithms Silver taught
us in the last 2 lectures. These include Monte Carlo Policy Iteration, TD_0 Policy Iteration, on-policy TD_Lambda/SARSA Policy Iteration,
and off-policy/Q_learning/SARSAMAX. You had successes with FrozenLake4x4
but had some issues with the TD methods for blackjack, as well as a harder time
with FrozenLake8x8. Although, all in all, I feel like I got a good understanding
of all the algorithms and how to implement them and their knobs to turn and the hyperparameters etc. 
But I can see the issue that as the state space evolves, the tabular methods won't be super helpful.
For one, the state spaces will just be too big, for two, there could be helpful information for nearby states you could pick up on without having to explicitly visit
that you won't get via tabular value/q-function methods.
Anyways, let's see what Silver has to say!

## Value Function Approximaters (no more tabular)

### Introduction
- There are two types of methods we will deal with
1) Incremental: With every state-action-reward combo we see we use that information
to update our value function approximation in this on-line manner.
2) Batch: Can use entire history, less online
This online/offline split isn't new, we've seen this before, but it can sometimes be confused
with the on-policy/off-policy distinction we have between something like SARSA/Q_learning.
On/Off-line refers to how often you update your Value/Q Function, on/off policy refers to whether behavior/target policies are the same!
- OK, why do we need value function approximation anyways?
Well think of backgammon, even, there are 10^20 states. Are you going to have a value for each and every one of these states??? Or game of Go... 10^170 states.
Hell no, you don't want to have to keep up with a value or even worse a Q dictionary for all 10^170 states and every associated action...
- We will be focusing on Value Function Approximation, not worrying about our policy functions right now.
That will come later.
- How we've been doing it so far:
Every state, or state_action has an associated entry in our V(s) or q(s,a)
This is a lookup table, and we update just only one specific s or (s,a) at a time.
- Problems with this approach:
Too many actions to store!
So slow if we only learn about one state/state_action at a time! Even with something like backwards TD which can allow us to update multiple, but still, takes too long.
- Solution: Turn this thing into an **actual** Value/Q FUNCTION, and not just a lookup table!
Our V(s) which only depended on the state and we could look it up, is now actually going to be a V(s, **w**), where **w** are the weights of our model!
Same with Q, q(s,a) --> q(s, a, **w**).
Given a state and action, we can use our model to map from these to our Q or V output!
This allows us to (as long as our model isn't HUGE) fit our value function in memory now! And also to generalize much better between nearby states and how
nearby states effect other's value mappings!
We are going to update the model, our model weights, using either TD methods or MC methods.
An MC method would take a whole episode, and cycle through the states using the G, while a TD method would bootstrap using itself and update the function every step.
- See Opex Daily 3-25-20 For some visuals
- The goal is again to approximate the TRUE value function or Q function, but we don't know the TRUE, so we use these models to get an estimate! Try and map tthis relationship.
- Which model? Which function approximater? 
Let's dig in to our machine learning toolkit:
Linear/logistic regression? Neural Nets? Decision Tree? Ensemble Trees? KNN?
Lots of choices!!
Well we want to pick choices in which it is quite simple to adjust our parameter vectors based on a new piece of info, we don't want to have to retrain from scratch every time!
This leaves linear/logistic regression, and neural nets as the best options.
These have gradients we can find, meaning how our output changes with respect to each of the parameters!
"The gradients with respect to each parameter are thus considered to be the contribution of the parameter to the error" - Random online blog I found

### Incremental Methods
- Using gradient descent, stochastic gradient descent because we update every step, with every new sample!
- See Opex Daily for math review and info on gradient descent and how we employ it!
 

