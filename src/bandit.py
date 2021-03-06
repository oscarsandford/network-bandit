import numpy as np
from dataclasses import dataclass

nruns = 100

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
			sds += [0.] * (num_states - len(sds))

		assert isinstance(means, list) and isinstance(sds, list) and len(means) == len(
			sds), "PeerArm: either means or sds is not a list, or they are and their lengths differ."

		self.states = [State(float(means[i]), float(sds[i])) for i in range(num_states)]

		if transmat is None:
			# Unspecified transmat means each state always transitions to itself. We can change this later if needed.
			self.transmat = np.eye(num_states)
		else:
			assert len(transmat.shape) == 2 and transmat.shape[0] == transmat.shape[
				1], "PeerArm: transmat must be two-dimension, and dimensions must be equal."
			assert transmat.shape[
					   0] == num_states, "PeerArm: transmat first dimension must equal number of peer states."
			self.transmat = transmat

		assert all([np.isclose(sum(self.transmat[i]), 1.0) for i in
					range(num_states)]), "PeerArm: a transmat row does not sum close to 1.0."

		self.current_state: int = 0

	def reset(self):
		"""
		Reset this arm's state to be the initial state it started with. 
		Called by environment at end of episode.
		"""
		self.current_state = 0

	def __str__(self) -> str:
		return f"State attrs: {self.states}\nTransmat:\n{self.transmat}"


"""

PEER GENERATION
What kind of environment layouts do we want? (i.e. number of peers, how many are "fast", how many are "slow"?)

"""


def create_peers(npeers: int, nstates_dist_fn, nstates_dist_params: dict, mean_dist_fn, mean_dist_params: dict,
				 sd_dist_fn, sd_dist_params: dict) -> list:
	"""
	A function to automate creating a bunch of peers.

	npeers: int                _ The number of PeerArms to create.
	nstates_dist_fn: fn        _ The function used to sample the number of states for a given peer (e.g. numpy.random.normal, numpy.random.poisson, ...).
	nstates_dist_params: dict  _ A dictionary with parameters for the nstates distribution.
	mean_dist_fn: fn           _ The function used to sample means for a peer's state (e.g. numpy.random.normal, numpy.random.poisson, ...).
	mean_dist_params: dict     _ A dictionary with parameters for the mean distribution.
	sd_dist_fn: fn             _ The function used to sample standard deviations for a peer's state (e.g. numpy.random.normal, numpy.random.poisson, ...).
	sd_dist_params: dict       _ A dictionary with parameters for the standard deviation distribution.
	"""
	peers = []

	for _ in range(npeers):
		nstates = int(nstates_dist_fn(**nstates_dist_params))
		if nstates < 1:
			nstates = 1
		means = []
		sds = []
		for _ in range(nstates):
			mean = mean_dist_fn(**mean_dist_params)
			sd = sd_dist_fn(**sd_dist_params)
			means.append(mean)
			sds.append(abs(sd))

		# Create transmat with random transition probabilities. (Optional?)
		transmat = np.zeros((nstates, nstates))
		for i in range(nstates):
			row = np.array([np.random.random() for _ in range(nstates)])
			row = row / np.sum(row)
			transmat[i] = row

		peer = PeerArm(means, sds, transmat)
		peers.append(peer)

	return peers


"""

ENVIRONMENT

"""


class BanditEnv:
	"""
	Create a bandit environment given a list of PeerArms.
	Contains methods to simulate the environment and return 
	peer-arm info.
	"""

	def __init__(self, arms: list):
		self.k = len(arms)
		self.arms = arms

	def reset(self, seed: int):
		np.random.seed(seed)
		# Reset current arm states to their initial state.
		for arm in self.arms:
			arm.reset()

	def step(self, action: int, timesteps: int = 1) -> list:
		"""
		Pull from a peer (arm) for timestep time. The reward is the
		total bytes the peer "receives" from this selection operation.
		Generate reward and then flip arm states for `timestep` times.
		"""
		assert -1 < action < self.k, "BanditEnv: invalid action."
		arm = self.arms[action]
		rewards = []

		# For each timestep, generate reward and transition all arms, because 
		# we are still moving through time, so the environment must change.
		for _ in range(timesteps):

			# Generating the rewards.
			reward = np.random.normal(arm.states[arm.current_state].mean, arm.states[arm.current_state].sd)
			rewards.append(reward)

			# Flipping the arm states for each time step.
			for a in self.arms:
				rand_state = np.random.choice(len(a.states), 1, p=list(a.transmat[a.current_state]))[0]
				a.current_state = rand_state

		return rewards

	def __repr__(self) -> str:
		return "bandit"


"""

ALGORITHMS

"""


def epsilon(strategy: str, arms: list, eps: float, nsteps: int = 10000, timesteps: int = 1) -> list:
	"""
	Epsilon-greedy and its variants: epsilon-first and epsilon-decreasing.

	strategy: str        _ Either "eps-first", "eps-decreasing", or "eps-greedy" - other values will default to the eps-greedy strategy.
	arms: list<PeerArm>  _ A list of PeerArm objects used in initializing the environment.
	eps: float           _ The eponymous hyperparameter.
	timesteps: int       _ The number of time steps to receive reward from a given arm. Can be made variable over time. Defaults to 1.
	
	Return a list of statistical results for plotting.
	We do 100 runs and 10000 rounds each run.
	"""
	global nruns

	env = BanditEnv(arms)
	steptotals = np.zeros(nsteps)

	for i in range(nruns):
		print(i, end=" ")  # PROGRESS MODE
		env.reset(i)

		Q = np.zeros(env.k)
		N = np.zeros(env.k, dtype=int)

		for t in range(nsteps):
			if strategy == "eps-first":
				# Epsilon-first: Always explore when below a certain threshold 
				# of rounds specified by epsilon * nsteps. In the paper, T is the 
				# number of steps, so we just use nsteps here instead of T.
				do_explore = t < (eps * nsteps)
			elif strategy == "eps-decreasing":
				# Epsilon-decreasing: Given the initial eps, divide by the 
				# current t+1 value to avoid division by zero.
				do_explore = np.random.random() < (eps / (t + 1))
			else:
				# Epsilon-greedy: Explore with probability eps, else greedy.
				do_explore = np.random.random() < eps

			if do_explore:
				a = np.random.choice(env.k)
			else:
				a = np.argmax(Q)

			rewards: list = env.step(a, timesteps=timesteps)

			# Once timesteps > 1 we need to average the rewards within "rewards".
			reward = np.mean(rewards)

			N[a] = N[a] + 1
			Q[a] = Q[a] + (1 / N[a]) * (reward - Q[a])

			steptotals[t] += reward

	return [total / nruns for total in steptotals]


def UCB(strategy: str, arms: list, C: float = 1, nsteps: int = 10000, timesteps: int = 1) -> list:
	"""
	arms: list<PeerArm>  _ A list of PeerArm objects used in initializing the environment.
	C: float             _ Controls the degree of exploration. C > 0. Defaults to 1.
	timesteps: int       _ The number of time steps to receive reward from a given arm. Can be made variable over time. 
							Defaults to 1.
	
	Upper Confidence Bound
	"""
	global nruns

	env = BanditEnv(arms)
	steptotals = np.zeros(nsteps)

	for i in range(nruns):
		print(i, end=" ")  # PROGRESS MODE
		env.reset(i)

		Q = np.zeros(env.k)
		N = np.zeros(env.k, dtype=int)

		for t in range(nsteps):
			# UCB algorithm starts
			ucb_vals = np.zeros(env.k)
			a = 0  # default action

			# go through each available action 
			for action in range(0, env.k):
				# making sure this action has been selected at least once to avoid divide by 0
				if N[action] > 0 and strategy == "UCB":
					ucb_vals[action] = Q[action] + C * (np.sqrt(np.log(t) / N[action]))
				elif N[action] > 0 and strategy == "UCB-1":
					ucb_vals[action] = Q[action] + C * (np.sqrt((2 * np.log(t)) / N[action]))
				else:
					ucb_vals[action] = 1e500  # make a large value so it is selected

			a = np.argmax(ucb_vals)  # select the best action

			rewards: list = env.step(a, timesteps=timesteps)

			# Once timesteps > 1 we need to average the rewards within "rewards"
			reward = np.mean(rewards)

			N[a] = N[a] + 1
			Q[a] = Q[a] + (1 / N[a]) * (reward - Q[a])

			steptotals[t] += reward

	return [total / nruns for total in steptotals]


def softmax(arms: list, tau: float, nsteps: int = 10000, timesteps: int = 1) -> list:
	"""
	SoftMax (Boltzmann Exploration)

	Arms are chosen according to a Gibbs distribution.
	Hyperparameter tau is called the "temperature".

	Arm a is chosen with probability p_a:

	p_a = \exp{Q[a] / tau} / \sum_{i=1}^k \exp{Q[i] / tau}
 	"""
	global nruns

	# For some reason, it seems that my input value for tau is "inverted" in its performance. Typically we 
	# want a low value of tau (e.g. 0.1), and so if I want the results expected of this heuristically 
	# optimal hyperparameter value, I have to pass 0.9 instead. This will not do, so this line below simply 
	# turns 0.1 into 0.9 for the mechanics of the algorithm, and allows the programmer to think he is clever. 
	tau = 1 - tau

	env = BanditEnv(arms)
	steptotals = np.zeros(nsteps)

	for i in range(nruns):
		print(i, end=" ")  # PROGRESS MODE
		env.reset(i)

		Q = np.zeros(env.k)
		N = np.zeros(env.k, dtype=int)

		for t in range(nsteps):
			# Gibbs probability distribution for arm according to SoftMax spec.
			probs = np.exp(Q / tau) / np.sum(np.exp(Q / tau))
			# SoftMax categorical draw.
			# (based on https://stackoverflow.com/a/62875642)
			cumlt_prob = np.cumsum(probs)
			a = np.argmax(cumlt_prob > np.random.random())

			rewards: list = env.step(a, timesteps=timesteps)

			# Once timesteps > 1 we need to average the rewards within "rewards".
			reward = np.mean(rewards)

			N[a] = N[a] + 1
			Q[a] = Q[a] + (1 / N[a]) * (reward - Q[a])

			steptotals[t] += reward

	return [total / nruns for total in steptotals]


def exp3(arms: list, gamma: float, nsteps: int = 10000, timesteps:int = 1) -> list:
	"""
	Exp3 (Exponential Weight Algorithm for Exploration and Exploitation) (Boltzmann Exploration)

	A variant of SoftMax.

	Arm a is chosen at time t with probability p_a(t):

	p_a(t) = (1-\gamma) (w_a(t) / (\sum_{j=1}^k w_j(t))) + (\gamma / k)
 	"""
	global nruns

	env = BanditEnv(arms)
	steptotals = np.zeros(nsteps)

	for i in range(nruns):
		print(i, end=" ")  # PROGRESS MODE
		env.reset(i)

		N = np.zeros(env.k, dtype=int)
		W = np.ones(env.k)

		for t in range(nsteps):
			# Exp3 weighted probability distribution for each action.
			probs = (1 - gamma) * (W / np.sum(W)) + (gamma / env.k)
			# Categorical draw.
			# (based on https://stackoverflow.com/a/62875642)
			cumlt_prob = np.cumsum(probs)
			a = np.argmax(cumlt_prob > np.random.random())

			rewards: list = env.step(a, timesteps=timesteps)

			# Once timesteps > 1 we need to average the rewards within "rewards".
			reward = np.mean(rewards)

			N[a] = N[a] + 1
			estimated_reward = 1.0 * reward / probs[a]
			W[a] = W[a] * np.exp(estimated_reward * gamma / N[a])

			steptotals[t] += reward

	return [total / nruns for total in steptotals]
