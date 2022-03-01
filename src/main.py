"""
Code snippets borrowed from an assignment. 
Will be changed later in ways appropriate for the project.
"""


import numpy as np 
import matplotlib.pyplot as plt
import gym
import random

from gym import Env, spaces

class DynBandit(Env):
	def __init__(self):
		# Define the observation space, there are five arms, each having two possible state.
		self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2),
				spaces.Discrete(2), spaces.Discrete(2)))
		  
		# Define an action space ranging from 0 to 4, 0: the first arm, ... 4: the fifth arm
		self.action_space = spaces.Discrete(5)
		
		# Sample from normal dists with these as means when choosing a reward.
		self.reward_means = [[0,1,2,3,1],[2,4,6,8,10]]
		# Arms are initialized in "high" (or 1) state (0 is low).
		self.arm_states = [1,1,1,1,1]
		
		
	def reset(self, episode_index):
		# An episode is over, initialization for running next episode
		## Important note: for each new episode, you must reset the random seed to a different value. 
		## Otherwise, your episodes are not independent. This is a common error in statistical inference-based learning. 
		## For example, you can use np.random.seed(time.time()) to avoid the problem. 
		
		# Passing the episode index as the seed guarantees each episode has a different seed, 
		# since np.random.seed requires an int, and casting time.time() to int could seed the
		# same int multiple times.
		np.random.seed(episode_index)
		# Reinitialize!
		self.__init__()
		
		
	def _get_obs(self):
		pass # We assume the bandit does not disclose state information. 
	
	def step(self, action):
		done = False
		# Assert that it is a valid action 
		assert self.action_space.contains(action), "Invalid Action"
		# apply the action, generate the corresponding reward, and update the state of the arm. 

		# Sample reward from appropriate arm (based on action) from Normal(mean=reward_mean, sd=1).
		reward_mean = self.reward_means[self.arm_states[action]][action]
		reward = np.random.normal(reward_mean, 1)
		
		# Markov process: 
		# - transition arm to different state (high=1, low=0), probability 0.4
		# - remains in current state otherwise (probability 0.6).
		if np.random.random() < 0.4:
			self.arm_states[action] = 0 if self.arm_states[action] == 1 else 1
		
		return [], reward, done, self.arm_states


def eps_greedy(eps: float) -> list:
	env = DynBandit()
	steptotals = [0]*10000
	for i in range(100):
		# Reset env with seed as the run iteration number. This guarantees each env seed will be different.
		env.reset(i)
		Q = [0,0,0,0,0]
		N = [0,0,0,0,0]
		for j in range(10000):
			p = np.random.random()
			if p < eps:
				a = env.action_space.sample()
			else:
				a = np.argmax(Q)
			
			obs, reward, done, states = env.step(a)

			N[a] = N[a] + 1
			Q[a] = Q[a] + (1/N[a])*(reward-Q[a])
			
			# print action, reward for TA to check if your reward function has been implemented correctly.
			# UNCOMMENT THE LINE BELOW TO SEE OUTPUT OF ACTION, REWARD, AND ARM STATES OVER TIME.
			# print(a, reward, states)
			
			# Record reward, store historical rewards for calculation
			steptotals[j] += reward
	
	env.close()
	# Calculate the final statistical result for each episode
	return [el/100 for el in steptotals]
