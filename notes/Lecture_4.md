# RL Course by David Silver, Lecture 4
## Notes by Nathan Franklin
## March-2020

- We are fresh off implementing policy evaluation/iteration and value iteration, and understanding how those work.
Those are examples of full information planning problems (not reinforcement learning problems) which solve both the
prediction and control problems in order to find optimal state/action value functions and policies.
- Now we dive in to some Model-Free Prediction algorithms, where nobody has given us the full MDP specification
but we still want to solve this problem! We still want the agent to figure out how to behave optimally!
- There are 2 major classes of this model-free, no perfect MDP specification type problem:
1) Monte-Carlo Learning:
Methods which go all the way to the end of a trajectory, and estimate the value function of a state
by looking at sample returns which we get on those trajectory samples
2) Temporal-Difference Learning:
Methods which just look one step ahead, instead of an entire trajectory, and then estimate the return, after one step.
3) TD(lambda)
Unifies these approaches, can go any intermediate number of steps along a trajectory to come up with our value function for our problem. 

### Introduction
- 
left off at 2:00 in lecture 4
