# Multi-Armed Bandits for Optimizing New Peers in Peer-to-Peer Networks

## Idea

Consider the setting of a peer-to-peer network wherein a new peer joins with the intent to be 
brought "up to speed" with the rest of the network as soon as possible. However, the new peer 
does not know the network speeds of its seeds, just how much data it receives over time when 
it chooses a peer and receives data from them for one time step. The reward is how many bytes 
received in that time slot. 

We want to be careful about defining the reward, because we want the agent to choose the peer 
that is transmitting the fastest. However, consider that network speeds may change, and the 
optimal seed to leech from will not always be the best. 

Various algorithms will be considered, starting with epsilon-greedy and UCB (upper confidence 
bound). More [here](https://en.wikipedia.org/wiki/Multi-armed_bandit).