# RL Course by David Silver, Lecture 5
## Notes by Nathan Franklin
## March-2020

### Introduction
- Alright we are back at it. We have been focused on model-free prediction methods, model free meaning
we don't have a model of the MDP, the transitions/rewards/etc before hand, and we have to estimate these things empirically from sample trajectories.
- Now we move to model-free control, which will use our model-free prediction to find some optimal policies hopefully!
- Why should we care so much about these unknown MDP reinforcement learning problems and algorithms?
Well because lots of actual interesting problems can be modeled in this form, while not many good/interesting
problems fit the full information MDP's we saw earlier.
- We have 2 paradigms in this world:
Before we dive in, let's establish a mental model. There are really 2 types of policies we can think of. The macro policy,
which is the policy we care about, it's the end-goal of our control/prediction loops. 
It is what we eventually play our game with. This is the long-term, important policy. 
Then there is the micro-policy, the thing which we actually follow to generate our sample trajectories and make our way around the state-space. 
Whether these 2 policies are the same object or different determine whether you are on- or off-policy!
1) On-Policy: The micro-policy which generates our trajectories is the same object as the macro-policy we want at the end. It self-improves
2) Off-Policy: The micro-policy, which we follow to generate our behavior and see states and stuff, is not the macro-policy which we care about overall.

### On-Policy Control

#### On-Policy Monte Carlo Control
- The main idea in these control methods is the same idea as in policy iteration which we used earlier! Please see Opex 3-8-20 saily diary for visuals/theory.
- But key is the fact that we aren't wedded to the prediction method of dynamic programming policy evaluation or simply a fully greedy control step.
Now we will tweak and eperiment with new algorithms for our prediction and control loops!
- One idea would just be, ok take policy_iteration algorithm we had before, and use MC policy evaluation for
the prediction step of evaluating a given policy and getting an updated value function with which to base our next policy on!
- Problems with this idea?
1) We won't explore the state space very much if our policy narrows early, we will deal with that later
2) The bigger issue for now is that our formula for the greedy policy control step previously used the MDP dynamics to its advantage. We no longer have that information!!
Instead, we now are going to pick our action not based on the state-value function, which needs the MDP model, but now our action-value function, the q(s).
See Opex book for this mathematically!
- So, we decided that we will actually estimate our action-value function from our monte-carlo sample trajectories,
and that will be returned from our prediction inner loop. Now what should we do for our evaluation step?

- Left off at 12:50, about to see if we can drop in q action estimatino in our evaluation step and just stay greedy?

