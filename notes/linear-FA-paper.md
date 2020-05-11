This is some notes for my own benefit, looking at a paper which compares various
RL approaches to solving cartpole. https://arxiv.org/pdf/1810.01940.pdf

This is sort of fun because you're really looking at academic research, 
and hopefully understanding everything they're talking about, thanks to
David Silver, and then going to implement it yourself! Enjoy Nate!!!!!

## Comparison of Reinforcement Learning Algorithms applied to the Cart-Pole Problem

- The goal of cartpole is to balance a pendulum in the upright position by using 
bi-directional force pushes imparted on the cart. 
- In the open-ai gym version we observe 4 things about the environment:
* Cart Position: [-2.4, 2.4]
* Cart Velocity: [-inf, inf]
* Pole Angle: [-41.8, 41.8]
* Pole Tip Velocity: [-inf, inf]
- We have 2 actions, 0 (push cart to left), and 1 (push cart to right)
- Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
- Reward is 1 for every step taken
- Because there are so many possible game-states, stemming from the continuous nature of the 4 dimensions we 
can find our cart-pole apparatus in, we will use function approximation. A tabular method would explode in size!
- A simple linear combination of the 4 available features with a feature parameter for each should suffice!
- This should achieve good results pretty quickly. They used alpha=0.07. gamma=0.992.
