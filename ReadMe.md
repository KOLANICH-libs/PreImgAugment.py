PreImgAugment.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
================
~~[wheel (GHA via `nightly.link`)](https://nightly.link/KOLANICH-libs/PreImgAugment.py/workflows/CI/master/PreImgAugment-0.CI-py3-none-any.whl)~~
~~[![GitHub Actions](https://github.com/KOLANICH-libs/PreImgAugment.py/workflows/CI/badge.svg)](https://github.com/KOLANICH-libs/PreImgAugment.py/actions/)~~
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH-libs/PreImgAugment.py.svg)](https://libraries.io/github/KOLANICH-libs/PreImgAugment.py)
[![Code style: antiflash](https://img.shields.io/badge/code%20style-antiflash-FFF.svg)](https://codeberg.org/KOLANICH-tools/antiflash.py)

This is a library of predefined image augmentation pipelines.

Currently it only makes sense to use `ImgPreImgAugment.autoaugment`. It is derived from `pytorch.autoaugment` (which is derived from autoaugment paper), but is quite different.

Originally it was implemented the following way:
1. there is a list of sequencies of 2 ops;
2. each op in a chain has 2 params, the intensity of the op from a fixd set of intensities and the probability of it being applied;
3. each chain is selected randomly (uniform distribution), then the op in the chain is applied with some probability.
4. those sequencies and params are recorded into a file.

Too many parameters, too little variability, only sequencies of length 2 are supported... My goal was a bit different from training neural networks, I just needed to transform an image in some sensible way.

Let's assumme that the distribution of sequencies of operations is independent from params of each op used. It allows us to model each one separately.

Instead of using those found sequencies of transforms of length `2` for each of datasets, I have just done the following.

To model the distribution of each op parameters:
1. Combined the found transforms for all the datasets into a flat list. Because between different datasets some ops params distribution were pretty consistentent.
2. Grouped by the operation.
3. Fitted the bivariate normal distribution to the params of each ops.
4. Those bivariate distribution were about correlation of probability of applying an op to its intensity.
5. But I guessed we need an univariate distribution of parameters only. So the fitted bivariate distribution has been transformed into 2 univariate distributions, one over intensities, another one over probabilities. Those distributions (now only probability one, I've made a mistake calculating the PDF over intensities) are different from the marginals of the joint because they used probability as weight, let's call them convolved ones. The convolved probability distribution has been integrated into a single number, its expectation. This probability is not used for now. The convolved distribution over values has been fitted with a gaussian, and its parameters have been recorded.
5. Saved the params of each op into a JSON file.

To model the distribution of sequencies I have utilized a 1st order Markov chain.

Then those 2 models are transformed into the stuff over `imgaug` framework:
* `imgaug` allows to specify ops which parameters are sampled from some distribution
* `imgaug` allows to compose ops into sequencies and call one with some probability

Tutorial is available as [`./tutorial.ipynb`](./tutorial.ipynb)[![NBViewer](https://nbviewer.org/static/ico/ipynb_icon_16x16.png)](https://nbviewer.org/urls/codeberg.org/KOLANICH-ML/PreImgAugment.py/raw/branch/master/tutorial.ipynb).
