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
- We call this present value of an entire trajectories reward the G.
- Be careful, G is this value over one single sampled trajectory. Not an expectation over a huge sample of trajectories. 
That expected value of G given a fully specified MRP will be important, but we aren't their yet!

You left off at 18:20 in lecture 2!

