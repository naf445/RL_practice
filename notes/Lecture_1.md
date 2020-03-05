# RL Course by David Silver, Lecture 1
## Notes by Nathan Franklin
## March-2020

### States and Markov Property
The state is defined as any function of the history of an agent/environment complex.
There are 3 different meanings of 'state' used by people.
- Environment State: the actual state of everything in the environment, perfect information about all the parts
of the environment and their corresponding full description 
- Agent State: The agent's internal representation of this environment state. Most of the time won't be identical
to the environment state and will be some form of subset and transformation of the environment state.
In fact you could choose your agent state to include a couple of the past state. It's flexible! 
We need to pick our agent state to be enough to predict the future! 
- Markov State: AKA information state. More mathematical definition. This definition thinks about state from
an information theory perspective and is formalized in that vein. This Markov State has a property called
the Markov Property. This property implies that all the information you need in order to make your best prediction
about the next state is contained by the current state. Having past states cannot help you. The current state
in a sense contains all the past information.
"The future is independent of the past, given knowledge of the present. If you have the Markov Property."

Not all state representation of things are Markov. For instance, if your agent is a helicopter, and your
state representation is simply the position of the helicopter, well you are not satisfying the Markovian
assumption, because you don't know in one time period, where the helicopter will be. To satisfy this Markov
property, you would need the velocity and windspeed etc.

Let's think about our Environment state and Agent state. Are they Markov? Well the environment state must be,
because if the environment couldn't predict the next time steps state based on the current, well you wouldn't 
have a functioning environment. By definition, the environment state is Markov. Yet the agent state, is not
necessarily Markov.

A key consideration in thinking about different reinforcement learning problems is "Does the agent state = environment state?"
If yes, this is a fully observable environment, fully observable by the agent. There are also partially
observable environments

### Models
A **model** in RL. Now this is a word which has confused you greatly in the past. The term model in RL
is used to mean a model belonging to an agent. It is something internal to the agent, which the agent uses to
predict the future states of the environment. 

Normally this model has 2 parts. A **transition model** and a **reward model**.
- transition model: predicting what the next state will be given the current
- reward model: predicting the reward in the next state

Using this type of model is completely optional, and many algorithms do not use a model at all. 

### Categorizing RL Agents
So now we've mentioned 3 key components: Policies, Value Functions, and Models.
We can now taxonomize all RL algorithms based on these 3 key components!

1) Value Based: Agent has a value function and the policy is just ad hoc, no set. Decision will just follow
directly from what the value function prediction tells it to. The agent doesn't need an explicit policy representation.
We have some function mapping the input states to the expected future rewards.

2) Policy Based: Instead of the agent having an internal estimate of the value from each state, we instead explicitly
represent and work with the policy. We have some function mapping from the input states to the action we should take.

3) Actor Critic: Combines the 2 methods together. This agent has an ability to map from states to an expected value
and from states to an action.

4) Model Free: The agent doesn't try and build it's own internal model simulating the environment.
Instead, the agent goes directly from state to policy or value function. No trying to take the states
and simulate the dynamics of the environment or the immediate next rewared.

5) Model Based: First thing we do with state inputs is build up our model of how the environment works, 
try and simulate either the next state or the immediate reward. Then we use that model to look ahead
and plan what step we should take.

So really these 3 key components, policy/value function/model, each RL algorithm is either a 1 or 0 in these 3 characteristics.
B/w these 3 things we have 8 categories of algorithms. 

### RL Dichotomies
RL Problems vs Planning Problems:
In RL problems, we don't know how the environment works. We don't know all of the rules and interactiona etc.
We have to use trial and error.
In a planning problem, we have access to a perfect simulation of the environment, we know all the rules. 
But we still have to figure out how to optimally play, choose the optimal action, find the optimal policy mapping from states to actions

Prediction vs Control:
- Prediction: Given a current policy, how much reward will I get if I follow it.
What is the value function (mapping of states to future rewards) given a set policy
- Control: Finding the best policy out of many possible policies.
Given a choice between a bunch of policies, which one is best.
- Normally we need to solve prediction in order to solve control problem.


