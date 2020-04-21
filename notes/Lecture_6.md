# RL Course by David Silver, Lecture 6
## Notes by Nathan Franklin
## March-April-2020

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
This on**line**/offline split isn't new, we've seen this before, but it can sometimes be confused
with the on-**policy**/off-policy distinction we have between something like SARSA/Q_learning.
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
Pick some loss metric for it and minimize your loss metric by tweaking parameters to "tune" it!
Our tabular V(s)s which only depended on the state and we look-up tables, now actually are going to be a full function with parameters, V(s, **w**), where **w** are the weights parameters/weights of this function we need to tune!
Same with Q, q(s,a) --> q(s, a, **w**).
Given a state and action, we can use our model to map from these to our Q or V output!
This allows us to (as long as our model isn't HUGE and has less parameters than total states) let our value functions be smaller than the lookup table versions
We can ideally encode the same amount of information now with less space, because we can take advantage of the relationships
between the parameters and how they add together to make our value estimation, we won't need a look up table!
Hopefully our value function even for a large state space can fit in memory now! And also to generalize much better between nearby states and how
nearby states effect other's value mappings!
We are still going to update the model/function, by tweaking our model weights/parameters, using either TD methods or MC methods so nothing much there has changed.
See past lectures for extensive breakdowns between these two aproaches!
- See Opex Daily 3-25-20 For some visuals
- The goal is again to approximate the TRUE value function or Q function, with these estimation methods
- What model should we user? What type of function and weights for this approximater? 
Let's dig in to our machine learning toolkit:
Linear/logistic regression? Neural Nets? Decision Tree? Ensemble Trees? KNN?
Lots of choices!!
Well we want to pick choices in which it is quite simple to adjust our parameter vectors based on a new piece of info, we don't want to have to retrain from scratch every time!
This leaves linear/logistic regression, and neural nets as the best options.
These have gradients we can find, meaning how our output changes with respect to each of the parameters!
"The gradients with respect to each parameter are thus considered to be the contribution of the parameter to the error" - Random online blog I found

### Incremental Methods
- Using gradient descent, and we update every step, with every new sample!
- See Opex Daily (4-16, originally you had it on 4-8 page but 4-16 is a much better version)
for math review and info on gradient descent and how we employ it!
- Also see Opex Daily (4-8-20) for review of some ways to think of feature vectors
- In a sense, the table lookup methods we learned prior are actually a special case of value function approximation!
It's just that the feature vector is a binary yes or no for EVERY SINGLE state we could find ourselves in!
This results in a feature vector as large as the state space!
And this "feature" is now just a binary indicator, are we in this state or not.
And if we think of our update rule now, we see that the feature value will be 1 while the others will all be 0, and thus only that weight will be updated!
- OK, look at this word incremental, doesn't that sound familiar to you?
YES, you've been dealing with incremental calculations this entire RL journey. We looked
at ways to calculate the mean incrementally, we use an incremental update for all our methods (SARSA, Q-learning, etc...) we've learned so far!
The commonalities to all of these methods are that instead of waiting to collect a bunch of information
about our target measurement, we update our estimate with every new piece of information.
We always have a base estimate, and then get some new estimate which we see how far away it is from that base estimate (and that gives us the direction)
of our error, and then we correct our base estimate in that direction by some alpha learning rate!
In these new function approximation methods, we are doing the same thing, but we are making it so every update is applied to ALL of the weights, in proportion to how active that feature was during the estimation.
All the weights will get updated by the same error, and same learning rate, but they will differ in their gradients, and that's where we update the weights individually!
- OK, but where are we going to get this "true" value of a state, so we can compare it to our base estimate and find the direction of our error for our
incremental update?
Well, the same way we've been doing it, using actual episodic experiences in either a monte carlo or TD fashion!
- See Opex Daily 4-16 for a couple good equations!
- See Opex Daily 4-20 for some notes on Value Function Approx
- See Opex Daily 4-20 for Control step w/ Value Function Approx
- The choice between MC methods, TD(0), TD(lambda), and the on-policy/off-policy versions are not trivial, this is for prediction/evaluation.
Sometimes they are good choices and sometimes they are not.
If you are using Table Lookup function approximation, they are all fine.
Linear function approximation, is fine for all on policy, but not for TD methods off-policy
Non-linear function approximation is not fine for TD methods. 
Of course this is all in theory and provable convergence, in practice sometimes these methods work fine!
- For control step, we find that we have no way to know for sure that our greedy policy improvement actually improves our policy in a fundamental way.
Our epsilon greedy step can be good, good, then bad, then good, then bad; we chatter around the optimal value policy and may never converge on it
 
### Batch Methods!
- So far we've talked about incremental gradient descent methods of weight updating, 
our updates just have us tweaking our weights a bit in the downhill direction.
But this is *not* sample efficient, we throw away every sample after we've used it and gotten out parameter gradient info out of it!
- Batch methods tries to make this process more efficient and make more use of all of our training points!
- One method would be to do a least squares method on all of our "training" data. 
Meaning all of our collections of states/actions/rewards.
- This is a different philosophy than the online gradient descent approach!
- There is an easy way to find this least squares batch solution --> Experience Replay
- Experience Replay
We collect all of our "training data", store it as we move through our trajectory.
Randomly sample from that data set, update using gradient descent!
This is different than the incremental approach which updated using every *newest* point, not a random sample from our past points.
Eventually, this gets us to the least squares solution actually. 
- Experience Replay Used in Deep Q-Networks (DQN)
This method combines 2 things we've seen before. 
1) Experience Replay: A batch method where we collect all our (state, value) pairs and cache them as training data to be sample from
2) Q-learning: Off-policy (trajectory generating policy is different from macro policy) policy iteration, where our target is given by the immediate reward plus the bootstrapped estimate of the State/Action we would choose given our policy.
Step 1) Take an action according to epsilon greedy 
Step 2) Store the SARS' in a replay memory D
Step 3) Randomly sample some minibatch from our replay memory D
Step 4) Compute our "labels" for all points in D, these labels are the Q-learning targets, the immediate reward plus the bootstrapped reward we would get with the S' from our policy
Step 5) Perform gradient descent to update parameters using these points and using MSE as the loss function
Another thing in DQN is that the targets we take as our "true" value, which we need to bootstrap, come from a value function which doesn't update every time, it is frozen for a bit!
They use a convolutional neural network to go from image --> parameters --> Q value
- Linear Least Squares Prediction (an alternative to sampling Experience Replay)
Using this experience replay, random sampling from our cache of SARS' amd using gradient descent,
it does find us this least squares solution eventually, but it may take a while!
But think back to econometrics or ML 101 and linear regressions, we could just closed form solution this with our batch of cashed data points! 
A little bit of linear algebra gets lets us skip to the correct solution!
With a small number of features, this is a reasonable approach to take!
Of course, we can heave Least Squares MC, TD-0, TD-L
- This approach manifests in Least Squares Policy Iteration
Evaluate by least squares Q-learning, improvement greedy
- Buckle up you have some fun coding ahead of you! Don't forget to deep copy!
