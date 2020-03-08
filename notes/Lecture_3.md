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

