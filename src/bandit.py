import numpy as np
import random

class BanditEnv:
	"""
	arms: int, the number of arms (possible peers) to try to connect to.
	attrs: list[2-tuple], a list of tuples for peer specification in the form
		(mean, sd) where `mean` and `sd` are the mean and standard deviation 
		to use for the reward distribution.
	"""
	def __init__(self, arms:int=5, attrs:list=None):
		assert all([len(a) == 2 for a in attrs]), "BanditEnv: attr specifications must be 2-tuples."
		self.k = arms
		self.means = [a[0] for a in attrs]
		self.sds = [a[1] for a in attrs]

	def reset(self, seed:int):
		np.random.seed(seed)
		self.__init__()

	def step(self, action:int, time:float):
		pass

	def __repr__(self):
		return "bandit"



