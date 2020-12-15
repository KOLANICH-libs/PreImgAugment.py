import numpy as np
from MatAndVecSer import *


class MarkovChainSample:
	__slots__ = ("parent", "p", "g")

	def __init__(self, parent):
		self.parent = parent
		self.p = self.parent.p0
		self.g = np.random.default_rng()

	def __iter__(self):
		c = self.g.choice(len(self.p), p=self.p)
		self.p = self.parent.T[c]
		yield c


class MarkovChain(MatAndVec):
	__slots__ = ()

	def __init__(self, p0: np.ndarray, T: np.ndarray) -> None:
		super().__init__(vec=p0, mat=T)

	@property
	def T(self):
		return self.mat

	@T.setter
	def T(self, v):
		self.mat = v

	@property
	def p0(self):
		return self.vec

	@p0.setter
	def p0(self, v):
		self.vec = v

	def __call__(self):
		return MarkovChainSample(self)


class MarkovChainParamsFiles(MatAndVecFiles):
	__slots__ = ()
	MAT_FILE_NAME = "T"
	VEC_FILE_NAME = "p0"

	MAT_AND_VEC_CTOR = MarkovChain
