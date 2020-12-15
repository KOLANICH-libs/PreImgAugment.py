import typing

import imgaug.augmenters as iaa
import imgaug.augmenters.contrast
import imgaug.augmenters.pillike as plk
import imgaug.parameters as iap
import numpy as np
from imgaug.augmenters.meta import Augmenter, OneOf

from ...AugmentersSpec import AugmentersSpec
from .augmenters import MakeAugmenters


def genAugsBank(augSp: AugmentersSpec) -> typing.List[Augmenter]:
	augsBank = [iaa.Identity()]
	names = iter(augSp.name2idx)
	firstEl = next(names)  # "Identity", but checked in other place

	for k in names:
		v = augSp.paramsDists[k]
		augFactory = getattr(MakeAugmenters, k, None)
		if augFactory is not None:
			# display(v)
			bothSigns = v.get("bothSigns", None)
			ps = v.get("params", None)
			if ps is not None:
				distName, loc, scale = ps
				dist = iap.Normal(loc, scale)
				if bothSigns:
					bDistName, bLoc, bScale = bothSigns
					if bLoc != 0:
						raise ValueError("For `bothSigns` `loc` must be 0!", bLoc)
					dist = iap.Choice([1, -1], replace=True, p=None) * dist
				augsBank.append(augFactory(dist))
			else:
				augsBank.append(augFactory(ps))
	return augsBank


class AutoAugment:
	"""Exponentially inefficient expansion of a Markov chain into a DAG of imgaug augmenters"""

	__init__ = ("augSp", "augmentersBank")

	def __init__(self, augSp: AugmentersSpec) -> None:
		self.augSp = augSp
		self.augmentersBank = genAugsBank(augSp)

	def genLayer(self, p: ndarray, T: ndarray, depth: int) -> OneOf:
		assert isinstance(p, np.ndarray)
		assert isinstance(T, np.ndarray)
		newDepth = depth - 1
		if newDepth:
			children = []
			for i, pp in enumerate(T):
				newLayerAug = self.augmentersBank[i]
				nextLayerAug = self.genLayer(p=pp, T=T, depth=newDepth)
				newLayer = iaa.Sequential([newLayerAug, nextLayerAug])
				children.append(newLayer)
		else:
			children = self.augmentersBank
		return iaa.OneOf(children=children, p=p)

	def getAugmenter(self) -> OneOf:
		return self.genLayer(p=self.augSp.mc.p0, T=self.augSp.mc.T, depth=self.augSp.usualLength)
