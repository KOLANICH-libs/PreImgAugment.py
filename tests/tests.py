#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest

thisDir = Path(__file__).parent

sys.path.insert(0, str(thisDir.parent))

from collections import OrderedDict

dict = OrderedDict

import PreImgAugment.autoaugment
from PreImgAugment.autoaugment import getAugmenter
from PreImgAugment.autoaugment.backends.imaug.utils import augment

from matplotlib import pyplot as plt
from PIL import Image


testFile = thisDir / "test.png"

testImage = Image.open(testFile)


class Tests(unittest.TestCase):
	def testSimple(self):
		aaug = getAugmenter()

		augmented = augment([testImage] * 10, aaug)


if __name__ == "__main__":
	unittest.main()
