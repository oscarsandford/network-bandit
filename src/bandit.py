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

		# A variable that can be changed based on an algorithm's measured reward of this arm.
		self.measured_reward = 0.

	def __repr__(self):
		return f"State attrs: {self.states}\nTransmat:\n{self.transmat}"


class BanditEnv:
	"""
	Create a bandit environment given a list of PeerArms.
	Contains methods to simulate the environment and return 
	peer-arm info.
	"""
	def __init__(self, arms:list):
		assert all([len(a) == 2 for a in attrs]), "BanditEnv: attr specifications must be 2-tuples."
		self.k = len(arms)
		self.arms = arms

	def reset(self, seed:int):
		np.random.seed(seed)
		self.__init__()

	def step(self, action:int, time:float):
		pass

	def __repr__(self):
		return "bandit"
