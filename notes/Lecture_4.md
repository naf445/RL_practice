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
- One thing to keep in mind here. These are all sort of impractical in interesting problems, because they learn the value function for 
EVERY SINGLE STATE. In reality, we will have far too complex of a state space to do things this way and we will have to learn
a more sophisticated way to approximate this state --> value mapping, using function approximation!

### Introduction
- So far we have used MDPs to formalize one specific type of problem, used dynamic programming to "solve"
these MDPs, by which we mean get an optimal value function and policy. This was for the planning problem. Full information
and fully specified MDP.
- We first used policy evaluation for the prediction problem of finding the value function of a given policy
and then used policy iteration with a greedy update to do the control problem and find the optimal policy
- But now we are going to be working with some problems where nobody is telling us exactly how the environment works!
We are shifting out of this highly impractical world of perfect information!
- However, the steps will be similar. We are going to first look at methods of policy evaluation in this imperfect knowledge world!
AKA focus on the prediction problem. We won't have someone telling us all the dynamics and transitions and rewards, we will have to figure them out.
- Eventually we will do control and try and optimize things, but let's dive in to the prediction problem first. Evaluating a given policy!

## Solving the Prediction Problem with Model Free (imprefect information MDP) Methods

### Monte-Carlo Learning
- Learn directly from episodes of experience.
- We don't need knowledge of MDP dynamics (rewards, transition probabilities), that stuff would constitute having a full "model"
- Look at complete episodes through our MDP, a full trajectory from initial state to termination
- Simplest possible idea. Your estimate of the value function is just the mean of the returns you actually get!
- This method is only approporiate to episodic MDPs, and also non-infinite ones
- It is possible to do all the necessary steps to solve the reinforcement learning problem with this simple technique!
- *Goal*: To find the value function (expected returns from that state onward) from any state in an environment, given a particular policy
Remember this is about evaluating a specific policy, and finding the value function given that policy
When we take a bunch of samples of a trajectory, we are going to get a buunch of trajectory histories
which consist of States, Actions, Immediate Rewards, States, Actions, Immediate Rewards, and so on and so forth...
Remember our friend G? He represents, for a single trajectory, the total discounted rewards from some specific state of our MDP
We originally defined the value of a state, given a policy, as the expected value of this G given we are in a specific state, but
expected value is only possible to get if we know everything about the environment, state transitions, etc. 
We now must replace this formulation of V to be the actual **empirical** mean we observe of all the discounted rewards for a given state.
- We can think of some problems here. How do we know our trajectories will even contain all states?
What if one is uncommon? What if we get unlucky on our only visit to that state? What if one trajectory has a state 1000 times and skews things?
- Below are some implementations of monte-carlo approaches which try and deal with some of these issues noted above
- First Visit Monte-Carlo Policy Evaluation / Prediction
1) Sample a bunch of trajectories
2) For every state, keep a running total of it's discounted future rewards and a count of how many times you've visited
but only count the first visit to an "s" in every trajectory. 
3) Calculate average of these G's for every state to get the V(s)
- Every visit MC Evaluation
Same as above but every visit to a state is counted in it's running reward total and visit count, even multiple times within a trajectory
- Let's just take a moment to realize the immensity of the problem we are asking our agents to learn. think in blackjack, your estimation
of future rewards depends on many things. Your strategy, the dealer's strategy, your hand, the dealers hand, the past hands played,
how many aces are left, how many high and low cards. There are tons of things, and instead of like in the full MDP problems,
we don't need to a-priori have access to all these possibilities and transitions and rewards. We have our agent learn them from sampling!
- As a math aside, it is actually possible to calculate a mean of a stream of numbers incrementally. We don't have to wait until we have
collected all of our sample trajectories to make an estimate of a V(s). See Opex daily diary for info
- Using this strategy, we can calculate our V(s)s incrementally, that is live with every trajectory, not getting the sums and counts
and doing a calculation at the end! Which brings us to a new version of a MC Prediction Algorithm -->
- Incremental Monte-Carlo Policy Evaluation
1) Sample a trajectory
2) Use incremental mean formula on every state visited to get a running V(s) for every state.
For this calculation, you will only need the past mean estimate, the current new value to add to the collection, and the collection size
3) Repeat
Keep in mind, the *incremental* part means we can update our value function for states after every trajectory.
We couldn't do it in the middle of a trajectory because we need to reach the end of the trajectory to be able
to get the Gs for all the states in that trajectory b/c we need to have future rewards for every state!!
Please see the opex daily diary for an explanation of the formula here as it is very helpful!
- Monte Carlo Learning Summary
So we saw what we do. Collect trajectories/episodes. Use these, either incrementally or at the end to calculate a V(s)

### Temporal-Difference Methods
- Similar to MC methods in that both learn from empirical sample of experience, both are model-free, don't need full MDP specification
- BUT, TD methods don't need FULL trajectories, they are ok with incomplete episodes, and use bootstrapping (which will estimate the rest of the episode from it's current guess of how things will go)
- They heavily use incremental algorithms, where a guess is updated by new information. Most of these are similar to the incremental mean
algorithm covered intuitively and mathematically in your Opex Daily Handbook
- Please also see Opex Handbook for mathematics of TD(0) algorithm which performs value function updates mid-trajectory!
- This may seem like it adds in more estimating and stuff on first glance and that the Monte Carlo version is better so you can sort of
see the ramifications of your actions. But think of a car driving scenario, if you nearly get in a wreck but then achieve your ultimate goal,
a MC approach you wouldn't end up with a crash, you wouldn't be able to update your value function at that state negatively. But with TD
approach, you would be able to know you were in a bad spot there!
In a sense, MC methods almost bias the states' value functions with information about the future they never would have had
and which it almost doesn't make sense for them to incorporate when they are estimating how good or bad a situation was.
It's like if we were about to crash the car, we don't want us to learn that we don't mind that situation just because eventually we were fine.
No we want the value function to not really take in to account the thing that happens way down the line quite as much.
Obviously there are caveats to this idea.
TD methods allow you to sort of increase the effect of local rewards/values and not allow the end reward to matter quite as much.

### MC vs TD:
- TD can learn a before a trajectory is over while MC must wait until the end of an episode
TD can learn even if trajectory never ends while MC couldn't
- In MC, using the entire trajectory, we actually get a statistically unbiased estimate of the value function.
But in TD learning, the TD target equaion includes our **best estimate of the value function** as one of the terms in it. Which is
different than the true value function. Thus this introduces bias in to the TD method.
While TD is more biased than MC, it does reduce the variance. With MC, think, your G for a given state now depends on your immediate reward,
and then all of these other states and events we may encounter which can wildly swing our final reward, making for a wide variance of
possible V(s)s. But for TD, you're only sampling one State --> action --> reward --> transition, so it's much less noisy.
- Summary
MC is higher variance, no bias. Good convergence properties. Not too sensitive to initial value. Simpler to understand and use
TD has low variance, some bias. Usually more efficient b/c lower variance. TD(0) does converge to true value function for a policy, most of the time, not w/function approximation.
TD is more sensitive to initial value.
- Let's dig in a little more to the intuitive approaches each of these algorithms take, and compare them.
MC converges to the solution which minimizes the mean squared error between the actual returns we've seen and our estimated value function.
TD: What TD is doing is it is implicitly actually building an internal model of the fully specified MDP with transition probabilities
and rewards and that sort of thing. Does this by just taking the mean of these values. See opex daily handbook for details/formulas.
- TD in a sense is exploiting the Markov Property, exploiting the assumption that the data stems from a Markov process, so by building
an internal model of this MDP, it can use it and it will be applicable.
MC methods don't exploit this Markov property. It has to blindly check out all the states of the trajectory and doesn't assume anything
even if they would be valid assumptions about the structure of the thing generating the data it's getting.
This is a good thing if you're in a non-Markov environment!
- Bootstrapping: Bootstrapping is the idea that we won't use the real returns but will just use our current V(s) as an estimate for future returns
MC does not bootstrap, it looks at entire trajectories and real returns
DP and TD both bootstrap. Think of the Bellman Equations, they include a term with the best estimate of our V(s). Same with TD
- Sampling: Are updates after a full exhaustive search or from a sample?
MC does sampling, TD does sampling.
DP doesn't do sampling. Does full exhaustive search before making an update.
- See opex daily handbook for visualization of this!

### TD Lambda: The spectrum between MC <---> TD(0)
- We have seen that both MC & TD(0) use sampling of dynamics to inform their update, unlike DP. But what they
do differently is the backup depth. MC does full backups while TD uses bootstrapping after seeing the imediate reward.
- There is actually a middle ground! --> TD(lambda)
- Essentially, we let the TD target look n_steps in to the future before we use our value function estimate!
- See Opex handbook for maths.
- But what n do we choose? We want to consider all of our N's and see what works best because sometimes it seems like
n=0 or n=1000 or something in between is best
- We can actually make a target which averages multiple different n-step answers for the value function.
- Our estimate of the return could be like 0.333 x V(n_steps=5) + 0.333 x V(n_steps=2) + 0.333 x V(n_steps=10000)
- So prior to this, we've been seeing TD(n_steps), but now with this averaging thing when implemented is called TD(lambda)
- And things aren't just weighted equally though. All the N_s aren't. They are weighted by a term which is a function of lambda. Given in Opex Handbook notes!
- This lambda however though if you think about it, we have to wait until an episode is done, because we need like v(n_50) and v(n_1000) and stuff.
This suffers from some of the same issues as MC, because it has to wait the whole time for an entire episode! It does bootstrap, but it uses all the N_s, so some of them bootstrap some of them dont...
- But, that is the forward TD(lambda), but now let's check backwards TD(lambda), where we still have the nice properties of TD learning, where it's
incremental, and we don't have to wait for an entire episode and we can get
some of the benefits of bootstrapping.
The key here is actually modifying our learning rate, and making it so that the TD error is just normal TD(1) error, it doesn't do all the TD(n_step) errors and average them by lambda weighting anymore.
The lambda does show up in the eligibility trace, which depends on 2 things: the frequency with which we visited the state, and the recency, and lambda!
- Eligibility trace: the eligibility trace is a function of the recency and frequency of something, in our case, of encountering a state.  
Every state get's an eligibility trace, and this affects the learning rate for this state's value function update. 


