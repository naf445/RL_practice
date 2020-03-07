# RL Course by David Silver, Lecture 2
## Notes by Nathan Franklin
## March-2020

### Markov Proccesses 
- Almost all RL problems can be formalized as an MDP, but the most basic element of this is the Markov Process,
which we detail now, before building on top of it in a little bit.
- Refresh on the Markov Property from Lecture_1.md notes. This is a key concept.
What happens next is only dependent on the current state.
- State transition matrix: gives us the probability that we transition from one state to the next states.
Records the probability for every single S0-S1 possibile transitions.
- Markov process is just a set of these Markov Property satisfying states (aka a state space, the set of states we can be in)
and an associated state transition matrix for all of these states!
- Once we have this complete markov chain specified, with all the states and their transition probs, we can start taking
some sample trajectories through this chain. And we can see what happens! We will get a collection of these random
sequences from this markov chain, each representing a path through the chain.

### The MRP
- Adding a reward in. Instead of fully specifying the problem/scenario by knowing the states and transition probabilities, 
we now need also to know the reward at every state!
- We can get an entire Markov Reward Chain's present value by summing together the rewards for every state traveled
on a trajectory through the chain, discounting each by a new term, gamma!
- We call this present value of an entire, specific trajectory reward the G.
- Be careful, G is this value over one single sampled trajectory. Not an expectation over a huge sample of trajectories. 
That expected value of G given a fully specified MRP will be important, but we aren't there yet!
- We also can speak of Value Functions with respect to MRPs. In particular, every individual state in a markov reward process
has an associated value of that state, which is the expected return if we were to find ourselves in that state.
- Now this value, V(s), is actually referring to an expected value now for a given state. So remember we had G, which was the
actual return we get from one trajectory, and then now V(s), is the expected G if we took a HUGE number of samples from
starting out in a specific S.
- **Bellman Expectation Equation for MRPs**: One of the most fundamental concept/relationship in RL!!!
But keep in mind the Bellman Equation we speak of here is the Bellman Expectation Equation, which is a bit different than the uber-important Bellman Optimality Equation
The Bellman Expectation Equation is just a way to think about the value function of a given state, that's all it is!
The idea of the Bellman Expectation Equation is that the Value function (which we just described as the expected value 
of G given a certain state S can be broken up into 2 key components:
1) The immediate & non-discounted reward of t+1
2) The Value Function of where you end up (which again is the expected G from that state)
In simpler terms, the overall value function of being in state S is the reward you get now + the expected reward of wherever you end up!
For the mathematics of this, which are actually not that confusing, please see your Opex desk diary in the entry on 3-6-20.
It's helpful to see how the Bellman Expectation Equation for MRPs naturally follows from our definitions of G and the Value Function as the expectation of G.
- So far we've seen that we could talk about every state's individual Bellman Equation, but there is a more concise
way to talk about the Bellman Equation as it relates to an entire MRP. This is using matrices and vectors. Again, see
Opex daily diary for mathematical details. This matrix form represents the Value Equation of an MRP for ALL of its states!
- Because the Bellman Equation is a linear eqn, it can be solved directly.
But this requires inverting a matrix, and this is very hard with large matrix and large state space, which
is why we resort to other methods to do this: dynamic programming, monte-carlo, temporal difference!
 
### The MDP
- While the MRP was a helpful buliding block, it is only a building block to the actural formalism
which we use in RL --> The Markov Decision Process (MDP)
- In the MRP, we were just put on a Markov Chain and we didn't have any decision to make, we just had
various probabilities with which we would transition states, and we were essentially carried through
the trajectory by whatever each state's transition probability matrix allowed us!
- Now, instead of just being carried through by whatever the transition matrix says, we now have an agent's action to consider.
- To fully specify an MDP, we need to know all the states, all the rewards at each state, and all the state-transition probability matrices for every possible action the agent can take!
This last bit is the key difference. Eseentially we have an individual MRP for every possible policy. So it's like the complexity of an MRP, but multiplied in complexity by the actions of the agent.
Notice this last part: It is an interplay between both the action on the agent's behalf and the standard transition-matrix 
which gives us our probabilities of where we transition to next. 
- This should make sense. Why would choose the MRP to formalize RL problems when in an MRP we
just flow through the trajectories based on the pre-determined transition matrix specified. No, the interesting problems
we want to get through are ones in which we agents have some agency, can control our transitions up to a point.
That is why MDPs are used as the base and formalism for RL problems!
- So now if we want to incorporate agent decisions into our MRPs and make them MDPs, we need a way to formalize these decisions!
- This is our **policy**, which is just what action an agent will take given it finds itself in a certain state!
- The reason we spent time on learning about the MRPs is because, when we fix our policy we are back to an MRP.
So we learned how to get probability matrix and reward vector for an MRP, and to get the equivalent structures
for an MDP, we just basically get the expected transition matrix and reward vector, which is the sum of the transition matrices
and reward vectors for every possible action weighted by what our policy tells us that actions probability is.
See daily diary for details.
- Now we've mentioned we can get MDP versions of the reward and state transition matrix which are basically the MRP versions
taken as a sum weighted by a policy's action probabilities, and now we can use these to get our MDP version of the value function!
- But, similar to the transition matrix and the rewards, now there isn't a single Value Function for an MDP. There is a single
value function for every specific policy! Once again, after we nail down the policy, then we have things like an MRP.
But we can probably tell that the equation for our Value Function is going to be the weighted sum of the value functions
for every action, weighted by the probability of those actions according to our policy. It only makes sense to speak of
value functions and rewards and transition matrices in the context of a given policy!
- Here we will introduce both the MDP state-value function, and also the MDP action-value function.
There is no analog to the q-function in an MRP, because in an MRP, we had no agency and were just carried through our path by probabilities.
- Now we can take our current definitions of the MDP version of v(s) and q(a, s), and again think about them in a way which
gets us to the MDP version of the Bellman Expectation Equations! Again, see daily diary for math stuff.
- The Bellman Expectation Equations really come down to 1 idea about how to figure out the Value of the current state that we are in.
It just say ok, how good our current spot it is the sum of the immediate reward combined with how good (the value) of the state we end up in!
- It's important to also think of the Bellman Expectation Equation for the action-value function, not only the state-value function.
This one says, ok the value of taking this action is just the immediate reward plus the action-value resulting from the policy of wherever I end up!
- Check Opex Desk Diary for some more helpful ways of thinking about these equations. The key takeaway is that while 
so far we have been thinking of the value function in terms of an immediate reward and then a next state value function, it's actually possible
to think of the value function in terms of the q-function. So
we can think of the state-value function in terms of the action-value function. And actually vice versa!! Opex diary has some good intuitions and math and visualizations here!
- OK, so we've learned some things about getting the MDP versions of state-value function and action-value function, which are
very helpful for us to be able to see what happens IF we behave in a certain way, IF we nail ourselves to a policy.
BUT, we want to know how to find policies which are best, how to compare policies once we know how to judge them!
- As a recap, in MRPs, there was no behavioral component, so solving it just meant learning how to best calculate the Value Function for a state. This new problem was not a concern whatsoever.
But in MDPs, to solve it, we need to find the best policy!
- So, while it is helpful to know how to evaluate state-value functions, this is only a tool to help us pick the best policies, the
best possible solution to the MDPs. 
- We are going to define a theoretical V_star(s), which is the maximum possible amount of value we can get from our current state. 
Said another but equally valid and important way, given that we've found an optimal policy out of all other policies, what is the value we will get from this state following that optimal policy?
- Similarly, q_star(s, a) is the maximum amount of reward we can get from starting in S and choosing action a.
Said another way, given that we have an optimal policy, what is the action-value we can get from the state that results after we commit to our action?
- Ok so we've been talking about optimal state-value functions and action-value functions, lets talk about Optimal Policies! Which would be great to know
- A policy is a mapping from states to actions. Which one is the BEST? 
- We should define our notion of BEST, of optimality though. We will say that a policy is relatively optimal compared to another policy,
if that policy's value function IN ALL STATES in the MDP is greater than or equal to the other policy's value function for those same states!
- How do we find this?
Well one way is, if you've got q_star, you've got a way through your MDP that gives you an optimal path.
- There is a theorem which tells us that for ANY MDP, there is always at least one policy we can find which is optimal. There can be multiple
but there will always be at least one.
Another theorem assures us that the state-value function or action-value function following an optimal policy will be the same
as the optimal state-value function or optimal action-value function. 
- So our big plan to find this optimal policy, is to solve for q_star! We are going to have to think how to do this.
- Intuitively, what we would do is like ok look at the 2nd to final state, and see what actions are available, and what their final rewards look like.
And then work your way backwards from there.
- So while we have been talking about the Bellman Expectation Equations, now we can talk about the Bellman Optimality Equations!
- See handy dandy Opex desk diary for intuition/math on the Bellman Optimality Equations!
 - Previously we could solve these Bellman Equations fairly simply with some matrix work, but actually these Bellman Optimality Equations
are much harder to solve! That makes sense because they are a bit more complicated than the Bellman Expectation Equations.
- We have no closed form solution, we have a maximum and need comparisons, so we resort to iterative solution methods.
- These are dynamix programming iterative methods (value iteration, policy iteration, Q-learning, SARSA)

