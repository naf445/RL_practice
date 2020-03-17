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
- Can we just drop in q-value prediction in our evaluation step and just stay greedy on control step?
Actually, this can lead to problem number 1. We will get stuck! In our full MDP problem using dynamic programming solutions, 
we were able to see all the posible states and actions and stuff. But from sampling, we aren;t
guarunteed to see any particular states, and acting greedily may make us too influenced by early "noisy" value estimations.
While you may have a found a local maximum value function choice, we want to make sure we do our best to find the global maximum!
- What's the answer to this issue???
A simple idea which actually works very well in practice! --> Epsilon-Greedy Control Step!
During our control step, instead of acting 100% greedy w.r.t. q-function from evaluation inner loop, we act epsilon-greedy.
See Opex book for deets! 3-17-20
- Great, so now we have re-outfitted our original policy improvement algorithm which we saw in full-information problems
to be able to help us solve these more interesting less than full MDP problems, using model-free methods. 
To do this we 
1) Swapped out the DP style policy evaluation of state-value functions with model-free q-value evaluation.
2) Further we swapped out the pure greedy improvement step with our new epsilon-greedy improvement strategy!
- Now this strategy is solid. But let's zoom in on this evaluation step. Remember from some lectures ago, 
that even after a few rounds of the inner evaluation loop, you have gotten a pretty good evaluation answer.
You don't have to do this ad infinitum on each inner prediction loop testing a certain policy.
We can actually get a lot of information from just a few evaluation inner loop runs and get a helpful policy jump just from that!
- Also, let's think, we want exploration from our epsilon-greedy definitely for a while, but for the final policy, 
the one we are "turning in to the teacher", we don't want this exploration aspect. We want it to be optimal, not needing to explore anymore.
We must come up with some decaying schedule for epsilon such that 2 conditions are met!
1) In the limit, we actually DO continue to explore. Epsilon never gets to 0.
2) In the limit, policy does become greedy! Epsilon becomes smaller and smaller
- If we put all this in to an algorithm --> G.L.I.E. Monte Carlo Control
1) Take a fully run episode
2) Go through every (s,a) combo present in the episode, we will be keeping track of how many times we have visitied every particular (s,a) combo.
3) Update the Q(s,a) for every (s,a) present in the episode, using the incremental mean formula we saw earlier. (this is more efficient than storing everything)
4) Use epsilon-greedy improvement strategy with a 1/episode_number decay schedule for epsilon
This algorithm is much more efficient if the evaluation step is only one episode sample and then you immediately improve your policy.

#### On-Policy TD Control & Policy Iteration
- Let's slot in TD learning instead of MC learning for the evaluation step in this iteration algorithm
- Stick with epsilon-greedy policy improvement
- We are also going to do this entire loop quite rapidly. The inner loop is going to evaluate ONE time step,
and then immediately send this q(S,a) function for an epsilon-greedy improvement step.
- Allows for rapid iteration
- Called SARSA! Which is really quite simple. Just a policy iteration algorithm with a single step TD learning inner loop evaluation and an epislon-greedy improvement step!!!!
See Opex Daily for some math and the update equation!
- We can also do what we did last time, think of an n-step version of td as our evaluation step. Maybe an n-step SARSA.
- We can even do a SARSA-lambda version, where we get our Q estimate by taking a weighted average over a bunch of different
n_step reward lengths, weighted by a lambda schedule. 
- And, like last time, we can actually to a SARSA-backwards-lambda, where instead of needing the whole episode to do forward lambda we can
use eligibliity traces and do a backwards lambda! This allows it to be a fully online algorithm again!
- See Opex Book for some of this fleshed out. 
- In SARSA base, we only can propogate information back by one step per episode.
But in SARSA lambda, we in effect can shout backwards to the q functions of earlier states
and update them with what we find in the future. Seems more efficient eh?

### Off-Policy Control
- Let's remember our mental model with the 2 policies. We have the micro policy being used to explore the state/action space, and generate sample trajectories,
and we also have the macro-policy, the end-goal, the result of all this iterating.
- So far, we have dealt with the case where the trajectory generating policy is the same as the macro policy, we have been doing On-Policy learning.
- We even have a name of this new policy, called Mu now, different from Pi, our macro policy.
- So the question is, how do we take the info we get from our micro policy, and use that to change our macro policy?
- One main method to accomplish this is through importance sampling --> see opex daily handbook
- We can try importance sampling with MC, not a great idea. Works better with td-lambda learning. See Opex book for some intuition on this.
- But, the smarter way to deal with this problem, not importance sampling, but Q-learning. Another way to solve the "off-policy problem"
- This is Q-learning, or SARSAMAX, or "Off-policy SARSA w/ epsilon-greedy behavioral policy and greedy target policy!"
Again, see Opex Daily for math and stuffs!

### Tying Things Off
We can compare the new methods we've learned where we use sampling in a non-full MDP information world
with the old methods of dynamic programming, which assumed a full MDP world. 
See Opex Daily for details. 
- One main point is that EVERY algorithm we've done is just a way to
get the best estimation of V(s) or Q(s, a) that we can,
in order to get a good policy to play the game!
- TD learning is just a way to sample empirically from an environment, and
try and use that to estimate our V(s) or Q(s,a). 
