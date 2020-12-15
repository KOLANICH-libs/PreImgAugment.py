import imageio
import numpy as np
from PIL import Image


def pil2imio(im):
	return imageio.core.util.Array(np.array(im.convert("RGB")))


def imio2pil(im):
	return Image.fromarray(im, mode="RGB")


def augment(imgs, aug):
	return [imio2pil(el) for el in aug(images=[pil2imio(el) for el in imgs])]
