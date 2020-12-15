import typing
from pathlib import Path

try:
	import mujson as json
except ImportError:
	import json

import numpy as np

from ...MarkovChain import MarkovChain, MarkovChainParamsFiles, MarkovChainSample


class AugmentersSpec:
	__slots__ = ("name2idx", "idx2name", "mc", "paramsDists", "usualLength")

	def __init__(self, acts: typing.Iterable[str], mc: typing.Optional[MarkovChain], paramsDists: typing.Mapping[str, typing.Mapping[str, typing.Any]], usualLength: int = 2) -> None:
		assert acts[0] == "Identity", "First element (index 0) must always be identity element"
		self.name2idx = {a: i for i, a in enumerate(acts)}
		self.idx2name = acts
		self.mc = mc
		self.paramsDists = paramsDists
		self.usualLength = usualLength

	@classmethod
	def loadFromParamsDists(cls, paramsDists: typing.Mapping[str, typing.Mapping[str, typing.Any]], markovChainFiles: None, usualLength: int = 2) -> "AugmentersSpec":
		exclusion = {"Shear", "Translate"}

		if markovChainFiles is not None:
			mc = markovChainFiles.load()
		else:
			mc = None

		return cls(
			acts=["Identity"] + [pn for pn in paramsDists if pn not in exclusion],
			mc=mc,
			paramsDists=paramsDists,
			usualLength=usualLength,
		)

	@classmethod
	def loadFromDir(cls, specDir: Path, usualLength: int = 2) -> "AugmentersSpec":
		paramsDists = json.loads((specDir / "dists.json").read_text())
		markovChainFiles = MarkovChainParamsFiles(directory=specDir)
		return cls.loadFromParamsDists(paramsDists, markovChainFiles, usualLength=usualLength)
