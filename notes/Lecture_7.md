Lecture 7 of RL David Silver

# Policy Gradient Methods

## Introduction:
- Just for my own mental model, the linear function I built before was for
predicting all 4 action_values, and this was a value function approximation method.
So right off the bat, this is different, because we are working on the policy directly, not simply
deriving our policy based on our value function!
- Policy Gradient: Adjust the policy weights in the direction which minimizes some loss function! Or maybe Gradient Ascent, we want to increase our rewrd function?
See Opex Daily from 4-16 for Gradient Descent refresher, this one is really good!!!
- Eventually we will get to Actor-Critic, which uses value function approximation from last class AND the 
policy function approximation we will be creating in this class!
- In policy function approximation, this function will be the actual thing which picks the action. 
The output of the function won't be q_values for the actions, it's an action itself! Or a distribution of actions to take with varying probability
- We are actually going to use gradient ASCENT, here. See Opex Daily 5-18 for a bit of a theory on this.
- Sometimes, policy representation is just more efficient. It's easier to just say move left, than to have to calculate the exact reason to move left!
- Sometimes, the step in Q learning or SARSA where we find the max Q value for an action is going to be
very costly, perhaps there's 50000 actions to choose from that we need to compute the max between all the
values. Directly working on the policy eliminates all these calculations
- But it really depends on the problem which one is better, and sometimes it's neither or a combination! Don't think of it like there's a black and white "better" way
- Why would we ever want a stochastic policy?
Iterative games, sometimes randomness is good so your opponent can't predict your behavior. Rock/Paper/Scissors, Poker, etc...
You could actually have scenarios with the EXACT same observation of the environment, but in reality, there are different optimal actions to take.
In this case, we can see that some stochastic policy will probably be necessary, or else we are guarunteed to be wrong in one of the cases...
This is called "state aliasing", when two different states have the same representation to an agent by the agent's imperfect observation
- State aliasing occurs when you have an imperfect information world, and also when the features your agent can observe are imperfect in capturing the environment! In this case, stochastic policy may be necessary
- See Opex 5-18 for some notes here!!!
- And Opex 5-26
- And Opex 6-4
- Actor critic is actually quite weird that it works if you think about it.
Basically, we had policy gradient, and said ok, we can implement policy gradient,
and move in a direction which is dictated by the "true" value function.
But then we replace this "true" value function with a value function estimate.
And actually, we can do this without introducing any bias at all! 
As long as certain conditions are met.

 
