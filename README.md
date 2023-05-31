
# rl_warmups

This repo focuses on deep reinforcement learning and contains some of the core pieces for doing basic rl..

I started this along my AI / ML journey but had to put it on pause for the time being. As such, it is untested and far from completed.


## Algorithms

### Deep Expected SARSA

 - TD (Temporal difference)
 - On-Policy
 - Supports experience replay buffer
 - Stochastic policy
 - Stabilize updates with a target network


To evaluate an Expected SARSA policy choose the action from a probability distribution weighted by the Q value of each action.


For control, you can calculate the target using the sum of the Q values of the next observation weighted by the probability of choosing each action.


### (DQN) Deep Q Network

 - TD (Temporal difference)
 - Off-Policy
 - Supports experience replay buffer
 - Deterministic policy
 - Stabilize updates with a target network

To evaluate a DQN policy, always take the action with the highest Q value.

For control, you can calculate the target using the maximum predicted Q value of the next state.


### REINFORCE

 - MC (Monte Carlo)
 - On-Policy
 - No replay buffer
 - Stochastic policy

To evaluate a REINFORCE policy choose an action from a probability distribution weighted by the Q values.

For control, collect an entire episode and unwind it, assigning rewards as appropriate.

The loss is the sum of the negative log probabilities of each action times the reward


### Actor Critic

 - MC (Monte Carlo)
 - On-Policy
 - No replay buffer
 - Stochastic policy
