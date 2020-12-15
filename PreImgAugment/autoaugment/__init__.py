def getAugmenter() -> "imgaug.augmenters.meta.OneOf":
	from .AugmentersSpec.specBundle import augSpec
	from .backends.imaug import AutoAugment

	augmenter = AutoAugment(augSpec).getAugmenter()
	return augmenter
