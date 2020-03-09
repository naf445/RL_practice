# RL Course by David Silver, Lecture 3
## Notes by Nathan Franklin
## March-2020

- We've been discussing how to formalize certain types of problems into MDPs
and how we think about MDPs and some of they key concepts like value-state function and action-state function.
We explored the concepts of optimal policies, optimal V and Q functions, but we didn't really talk about how to find them!
aka How to build things that can solve the problem of finding these things.
- So we now introduce the concept of Dynamic Programming, which is how we're going to go about solving things!

### Dynamic Programming
- We are going to start with some ideas called policy evaluation, policy iteration, value iteration. And we can think of these as simple MDP solving methods.
And then from these building blocks, we will add in some wrinkles that allow us to develop these building blocks in to
more complex and general Reinforcement Learning methods.
- Why is it called dynamic programming?
Dynamic: Dealing with problems involving some sequential or temporal component
Programming: In the mathematical sense, like linear programming uses the term. It refers to the function which maps states to actions, the policy.
Dynamic programming is just a method which can be used to solve problems that satisfy a few conditions
1) They are composed of sub problems
2) These sub problems overlap and have similar solutions. Recurring sub problems
It just so happens that MDPs satisfy both of these conditions, making them good candidates to be solved by dynamic programming.
The Bellman equations, which define value-functions in a recursive manner, exemmplify this. Positive value functions in the form
of the Bellman equation is basically the reason why we can use DP to solve MDPs!!
- We are going to try to use DP for our "planning" problem. Planning is just part of the RL problem.
It is the part that assumes you have perfect knowledge of the MDP, so you know how the environment works and the rewards and transitions etc.
Specifically, it is asking us to be able to compute the value function for an MDP.
See https://github.com/naf445/RL_practice/blob/master/notes/Lecture_1.md for breakdown of RL vs planning problem.
- Now there is a dichotomy of prediction vs control, also discussed in that lecture_1.md notes. 
Prediction is given a policy, what is the value function?
Control is how to find the best policy once we know how to evaluate well with prediction.
- We will use DP to solve the prediction task and control task within this planning problem. 

### Policy Evaluation: Solve the prediction problem
- The name policy evaluation can be confusing, because sometimes it seems like we are evaluating a value function. 
In reality, we are focusing on a value function, but for a SPECIFIC policy. We are only looking at the
value function as a way to evaluate the policy we are interested in!
- Given a fully specified MDP states and environment and rewards and transitions etc., and given a policy, what is the value function?
- How will we do this? Apply the Bellman Expectation Equation for MDPs. 
- Bellman Expectation Equation is used for prediction/evaluation, whereas optimality version of Eqns comes in later during Value Iteration 
- We are going to take our Bellman Expectation equation, and iterate upon it continually, updating our estimate each time!
1) We will start off with some random or arbitrary initial value function, specified for every single state --> v1
2) Do a one step look ahead at all the states, and get a new, updated value function --> v2
- We will use *synchronous backups* to do this.
Every update we will loop over every single state in our MDP, giving us an entirely new value function from the time step prior
- See opex daily diary handbook for some good stuff here, 3-8-2020
- So basically policy evaluation solves the prediction problem by iterating on a random/arbitrary initial value function using the Bellman Expectation Equations and eventually is guarunteed to converge! That is pretty cool.
- To jump ahead a bit, we have focused on the prediction problem, but maybe we can think a little and see how it helps us with the control problem.
From this process of iteratively updating our value function, we can actually glean what a better policy might have been. 
Maybe we could just update our policy to behave optimally with respect to every new value function estimate's resulting greedy policy. --> Policy Iteration 

### Policy Iteration: Use policy evaluation/prediction to help us with the control problem!
- In policy evaluation, we focused on getting the v(s) for a given policy. Now we want to use this skill to solve the control problem.
1) Use policy evaluation(which is itself iterative) to get a v(s) for a generic/arbitrary/random policy.
2) Change our initial policy to be optimal w/ respect to this value function.
3) Use policy evaluation to get the v(s) for this new policy!
- This process does always converge to the optimal policy! But it may take a while...
- Mental Model: See opex daily handbook! This is a visualization of our process to solve these things
- There is a mathematical proof showing that if we pick our new policy by acting greedy based on our most recent prediction/policy evaluation step
then our value from this new policy is at least as much as the old policy.
- Because we know we aren't making our policy worse, if we stop getting improvement, we have found the maximum/optimal policy and
satisfied the Bellman Optimality Equation meaning we've also found an optimal policy!
- Summary so far!!
We started off with a way to evaluate, given a specified policy and an MDP, the value function for every state in that MDP.
We did this using an iterative approach with the one step look-ahead method using the MDP Bellman Expectation Equations
Then we found a way to use this new value function, and improve our current policy. And then we put these in an iterative loop.
There is actually a mathematical proof showing that this double iteration process does actually find us an optimal value function and thus an optimal policy!
- Now this process may take forever or a long ass time to get the optimal value function, and it's possible we achieve an optimal
policy long before. So we can introduce some stopping condition, like an amount the value
function needs to change by or some maximum number of iterations before we say we have found a new V(s) and update our policy greedily or something and do a early stopping.

### Value Iteration: Another mechanism to solve the planning problem!
- Our goal once again is to find an optimal policy! 
- But this time, we are going to focus on trying to find v_star
- Instead of starting with a policy and iteratively applying Bellman Expectation Equation to an arbitrary initial value function, we
are going to still start with an initial arbitrary optimal value function, but this time we are going to follow directions
by taking the maximum for all actions of the quantity R(s,a) + discounted weighted sum of possible s' value functions (using current v as best guess of v_star)
from the Bellman Optimality Equation, and follow what IT says to do, which is updates this optimal value function (V_star)
- We really don't bother ourselves with updating some policy and then evaluating the value function for that policy, we just replace our stand-in optimal value function
with the optimal value function we get after looking at all the actions allowed.
- We do this in a **synchronous** manner, meaning we fix our value function at each state as the current best estimate
for the optimal value function, and then go through every state and update the value function according to the Bellman Optimality Eqn
at every state. Only once we have done this for every state
do we consider these states NEW optimal value function to be the one we just found.
1) Assign an arbitrary, stand-in optimal v_star function to every state in our MDP
2) Look to the Bellman Optimality Equation to see what to do to perform an update and get a new estimate of the v_star,
for every state. Do this synchronously for every state before assuming we have a new stand-in v_star.
- Maybe you're wondering why you use the Bellman Optimality Equation for value-iteration but the Bellman Expectation Equation for Policy Evaluation.
Especially because it seems like we posit the same thing at the beginning, an arbitrary value function, right??? 
The key lies in a distinction between these initial value functions. In ploicy iteration/evaluation, we assume that we have just an initial arbitrary regular value function.
For Value iteration, we assume we have an initial, arbitrary v_star, an optimal value function! This is
the key distinction and is why we use bellman optimality vs bellman expectation!!!
- Interestingly, this process is actually equivalent to doing policy iteration, but only allowing one run of the policy evaluation loop, 
only allowing you to update your v function one time.

### Summary of DP So Far for the Planning MDP Problems we've seen!!
If you get the below table, you're doing well and you're on the right track!!

| Algorithm | Problem it Solves | Bellman Eqn it Uses |
| ------------- |:-------------:| -----:|
| Iterative Policy Evaluation  | Prediction (given a policy, tell me the value function for all the states)  | Bellman Expectation Equation  |
| Policy Iteration  | Control (actually give me the best policy)  | Bellman Expectation Equation within the policy evaluation step, then greedy policy improvement after that  
| Value Iteration  | Control  | Bellman Optimality Equation  |

- All of these are based so far on state-value functions. We could actually do action-value function versions as well
- They are a bit more complex, but later we will see this is a good idea in some situations!

### Extensions
- In practice, doing these synchronous updates may be overkill. You can often
pick one state to be how you update your value function or v_star, whether in policy evaluation or value iteration respectively, and not have to
go through EVERY SINGLE state before announcing your new value func or v_star.
- There are ways to make this choice of which state or states to update the value function at: In-place DP, Prioritized sweeping, Real-time DP
- Eventually, for these methods, the curse of dimesionality is going to be so great
we are going to need some way to get around the planning problem conditions
where we have to know the fully specified MDP conditions and everything. In order to transition to
the reinforcement learning problem, we are going to instantiate some more efficient sampling methods, 
which are model free in nature, and don't require knowing the MDP model fully!
