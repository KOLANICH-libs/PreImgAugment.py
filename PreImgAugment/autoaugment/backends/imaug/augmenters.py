import typing

import imgaug.augmenters as iaa
import imgaug.augmenters.contrast
import imgaug.augmenters.pillike as plk
import imgaug.parameters as iap
import numpy as np
from imgaug.random import RNG

MagT = typing.Union[float, iap.StochasticParameter]


class ArcTan(iap.StochasticParameter):
	def __init__(self, val: iap.Multiply) -> None:
		super(ArcTan, self).__init__()
		self.val = val

	def _draw_samples(self, size: typing.Tuple[int], random_state: RNG) -> np.ndarray:
		samples = self.val.draw_samples(size)
		return np.arctan(samples)


interpolation: int = 3  # order, 0 = nearest neighbour
fill: int = 0
wrapMode = "wrap"  # "constant"


class MakeAugmenters:
	def ShearX(magnitude: MagT) -> iaa.ShearX:
		return iaa.ShearX(
			shear=ArcTan(magnitude) * 180 / np.pi,
			order=interpolation,
			cval=fill,
			mode=wrapMode,
		)

	def ShearY(magnitude: MagT) -> iaa.ShearY:
		return iaa.ShearY(
			shear=-1 * ArcTan(magnitude) * 180 / np.pi,
			order=interpolation,
			cval=fill,
			mode=wrapMode,
		)

	def TranslateX(magnitude: MagT) -> iaa.TranslateX:
		return iaa.TranslateX(
			percent=magnitude,
			order=1,
			cval=0,
			mode=wrapMode,
		)

	def TranslateY(magnitude: MagT) -> iaa.TranslateY:
		return iaa.TranslateY(
			percent=magnitude,
			order=1,
			cval=0,
			mode=wrapMode,
		)

	def Rotate(magnitude: MagT) -> iaa.Rotate:
		return iaa.Rotate(
			rotate=-1 * magnitude,
			order=interpolation,
			cval=fill,
			mode=wrapMode,
		)

	def Brightness(magnitude: MagT) -> plk.EnhanceBrightness:
		return plk.EnhanceBrightness(factor=1.0 + magnitude)

	def Color(magnitude: MagT) -> plk.EnhanceColor:
		return plk.EnhanceColor(factor=1.0 + magnitude)

	def Contrast(magnitude: MagT) -> plk.EnhanceContrast:
		return plk.EnhanceContrast(factor=1.0 + magnitude)

	def Sharpness(magnitude: MagT) -> plk.EnhanceSharpness:
		return plk.EnhanceSharpness(factor=1.0 + magnitude)

	def Posterize(magnitude: MagT) -> iaa.Posterize:
		"""Similar, but different"""
		return iaa.Posterize(
			nb_bits=magnitude,
			# interpolation='linear',
			interpolation="nearest",
		)

	def Solarize(magnitude: MagT) -> iaa.Solarize:
		return iaa.Solarize(
			p=1,
			threshold=magnitude,
			invert_above_threshold=True,
		)

	def AutoContrast(magnitude: MagT) -> plk.Autocontrast:
		return plk.Autocontrast(cutoff=0)

	def Equalize(magnitude: MagT) -> plk.Equalize:
		return plk.Equalize()

	def Invert(magnitude: MagT) -> iaa.Invert:
		return iaa.Invert(
			p=1,
			per_channel=False,
			min_value=None,
			max_value=None,
			threshold=None,
			invert_above_threshold=0.5,
		)

	def Identity(magnitude: MagT):
		return iaa.Identity()
