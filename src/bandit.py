import numpy as np
import random
from dataclasses import dataclass


@dataclass
class State:
	mean: float
	sd: float


class PeerArm:
	"""
	Specification for reward distributions of a single peer (or "arm" in MAB context) in 
	the network. You can specify this multiple ways:
	
	Create a peer with three states and three mean and standard deviation attributes.
	>>> PeerArm([4.3, 7, 8], [0.1, 0.5, 2.7])
	State attrs: [State(mean=4.3, sd=0.1), State(mean=7.0, sd=0.5), State(mean=8.0, sd=2.7)
	
	A peer with three states and lacking standard deviation attributes. Pads with 0.0's.
	>>> PeerArm([4.3, 7, 8], [0.1])
	State attrs: [State(mean=4.3, sd=0.1), State(mean=7.0, sd=0.0), State(mean=8.0, sd=0.0)]	

	A single-state arm with a single mean-sd attribute pair.
	>>> PeerArm(4.3, 0.5)
	State attrs: [State(mean=4.3, sd=0.5)]
	
	A single-state arm with a single mean and no defined sd. Defaults to 0.0.
	>>> PeerArm(4.3)
	State attrs: [State(mean=4.3, sd=0.0)]
	
	A two-state arm with defined transmat.
	>>> PeerArm([4.3, 7], [0.1, 0.5], np.array([[0.6,0.4] ,[0.7,0.3]]))
	State attrs: [State(mean=4.3, sd=0.1), State(mean=7.0, sd=0.5)]
	Transmat:
	[[0.6 0.4]
	 [0.7 0.3]]
	
	In an instance of this class, you can access the arm attributes 
	for each state, with `i` as the index of the state.
	>>> arm.states[i].mean
	>>> arm.states[i].sd
	"""
	
	def __init__(self, means, sds=None, transmat=None):
		if isinstance(means, list):
			num_states = len(means)
		else:
			means = [float(means)]
			num_states = 1

		if sds is None or (isinstance(sds, list) and len(sds) == 0):
			sds = [0. for _ in range(num_states)]
		elif isinstance(sds, float) or isinstance(sds, int):
			sds = [float(sds)]

		if len(sds) < num_states:
			sds += [0.] * (num_states-len(sds))

		assert isinstance(means, list) and isinstance(sds, list) and len(means) == len(sds), "PeerArm: either means or sds is not a list, or they are and their lengths differ."

		self.states = [State(float(means[i]), float(sds[i])) for i in range(num_states)]
		
		if transmat is None:
			# Unspecified transmat means each state always transitions to itself. We can change this later if needed.
			self.transmat = np.eye(num_states)
		else:
			assert len(transmat.shape) == 2 and transmat.shape[0] == transmat.shape[1], "PeerArm: transmat must be two-dimension, and dimensions must be equal." 
			assert transmat.shape[0] == num_states, "PeerArm: transmat first dimension must equal number of peer states."
			self.transmat = transmat

		# # A variable that can be changed based on an algorithm's measured reward of this arm.
		# self.measured_reward:float = 0.
		self.current_state:int = 0

	def reset(self):
		"""
		Reset this arm's state to be the initial state it started with. 
		Called by environment at end of episode.
		"""
		self.current_state = 0

	# def __repr__(self):
	# 	return f"State attrs: {self.states}\nTransmat:\n{self.transmat}"


"""
What kind of environment layouts do we want? (i.e. number of peers, how many are "fast", how many are "slow"?)

"""

class BanditEnv:
	"""
	Create a bandit environment given a list of PeerArms.
	Contains methods to simulate the environment and return 
	peer-arm info.
	"""
	def __init__(self, arms:list):
		self.k = len(arms)
		self.arms = arms

	def reset(self, seed:int):
		np.random.seed(seed)
		# Reset current arm states to their initial state.
		for arm in self.arms:
			arm.reset()

	def step(self, action:int, timesteps:int=1) -> list:
		"""
		Pull from a peer for timestep time. The reward is the
		total bytes the peer "receives" from this selection operation.
		Generate reward and then flip arm states for `timestep` times.
		"""
		assert -1 < action < self.k, "BanditEnv: invalid action."
		peer = self.arms[action] 
		rewards = []
		
		# For each timestep, generate reward and transition all arms, because 
		# we are still moving through time, so the environment must change.
		for _ in range(timesteps):
			
			# Generating the rewards 
			reward = np.random.normal(peer.states[peer.current_state].mean, peer.states[peer.current_state].sd)
			rewards.append(reward)
			
			# Flipping the arm states for each time setp
			for i, arm in enumerate(self.arms):
				rand_state = np.random.choice(len(arm.states), 1, p=list(arm.transmat[arm.current_state]))[0]
				self.arms[i].current_state = rand_state

		return rewards


	def __repr__(self) -> str:
		return "bandit"


def epsilon(strategy:str, arms:list, eps:float, timesteps:int=1) -> list:
	"""
	strategy: str        _ Either "eps-first", "eps-decreasing", or "eps-greedy" - other values will default to the eps-greedy strategy.
	arms: list<PeerArms> _ A list of PeerArm objects used in initializing the environment.
	eps: float           _ The eponymous hyperparameter.
	timesteps: int       _ The number of time steps to receive reward from a given arm. Can be made variable over time. Defaults to 1.
	
	Return a list of statistical results for plotting.
	We do 100 runs and 10000 rounds each run.
	"""

	nruns = 100
	nsteps = 10000

	env = BanditEnv(arms)
	k = len(arms)
	steptotals = np.zeros(nsteps)
	
	for i in range(nruns):
		env.reset(i)

		Q = np.zeros(k) 
		N = np.zeros(k, dtype=int)

		for t in range(nsteps):
			if strategy == "eps-first":
				# Epsilon-first: Always explore when below a certain threshold 
				# of rounds specified by epsilon * nsteps. In the paper, T is the 
				# number of steps, so we just use nsteps here instead of T.
				do_explore = t < (eps * nsteps)
			elif strategy == "eps-decreasing":
				# Epsilon-decreasing: Given the initial eps, divide by the 
				# current t+1 value to avoid division by zero.
				do_explore = np.random.random() < (eps / (t+1))
			else: 
				# Epsilon-greedy: Explore with probability eps, else greedy.
				do_explore = np.random.random() < eps

			if do_explore:
				a = np.random.choice(env.k)
			else:
				a = np.argmax(Q)
			
			rewards:list = env.step(a, timesteps=timesteps)
			
			# Once timesteps > 1 we need to average the rewards within "rewards".
			reward = np.mean(rewards)

			N[a] = N[a] + 1
			Q[a] = Q[a] + (1/N[a])*(reward-Q[a])
			
			steptotals[t] += reward
		
	return [el/nruns for el in steptotals]


def UCB(): 
	"""
	Upper Confidence Bound
	"""


	pass


def POKER():
	"""
 	Price of Knowledge and Estimated Reward (POKER)

	"""

	pass
