import re
import typing
from collections import defaultdict
from copy import deepcopy
import sys
from pathlib import Path

import mujson as json
import numpy as np
import scipy.stats
import StatsUtils.dists.Normal  # pylint:disable=unused-import
import StatsUtils.dists.TruncatedNormal  # pylint:disable=unused-import
import torchvision.transforms.autoaugment as ta
from icecream import ic
from scipy.integrate import simpson
from scipy.optimize import differential_evolution
from scipy.stats._multivariate import multivariate_normal_frozen
from StatsUtils.dists.MultivariateGaussian import MultivariateGaussian
from StatsUtils.plot import plotPrecomputedHistogram
from StatsUtils.utils import BoundsT, MultivariateBoundsT, calcRealMean, computeFunctionOnGrid


thisDir = Path(__file__).parent
repoRootDir = thisDir.parent

sys.path.insert(0, str(repoRootDir))

from PreImgAugment.autoaugment.AugmentersSpec import AugmentersSpec
from PreImgAugment.MarkovChain import MarkovChain, MarkovChainParamsFiles

# pylint:disable=protected-access,too-many-arguments,too-many-instance-attributes,too-many-locals

intensitiesBins = 10

NumT = typing.Union[int, np.int32, float, np.float32, np.float64]

ActionParamsT = typing.Tuple[str, NumT, typing.Union[int, None]]
ChoicesT = typing.Iterable[ActionParamsT]
PolicyT = typing.Iterable[ChoicesT]
PoliciesT = typing.Mapping[str, PolicyT]


def getPolicies() -> PoliciesT:
	return {p.name: ta.AutoAugment._get_policies(None, p) for p in ta.AutoAugmentPolicy}


ParamSpaceT = typing.Union[typing.Tuple[NumT, NumT], bool, None]
ParamSpaceSpec = typing.Mapping[str, typing.Iterable[ParamSpaceT]]


def getParamsSpaces() -> ParamSpaceSpec:
	sp = ta.AutoAugment._augmentation_space(None, intensitiesBins, [1, 1])
	res = {}
	for k, (rng, isSigned) in sp.items():
		rng = rng.numpy()
		if not rng.shape or rng.shape[0] == 1:
			rng = None
		else:
			rng = (rng[0], rng[-1])
		res[k] = [rng, isSigned]
	return res


def getFromLinspace(start: NumT, stop: NumT, count: int, i: int) -> NumT:
	return start + (stop - start) * i / (count - 1)


def convertSingleTransformPipeline(pp: ChoicesT, paramsSpaces: ParamSpaceSpec) -> ChoicesT:
	res = []
	for (op_name, p, intensityBinNo) in pp:
		params, _signed = paramsSpaces[op_name]
		if params is not None:
			start, stop = params
			if intensityBinNo is not None:
				val = getFromLinspace(start, stop, intensitiesBins, intensityBinNo)
		else:
			val = None

		res.append((op_name, p, val))
	return tuple(res)


def convertSinglePolicy(p: PolicyT, paramsSpaces: ParamSpaceSpec) -> typing.Tuple[ChoicesT]:
	res = []
	for tp in p:
		res.append(convertSingleTransformPipeline(tp, paramsSpaces))
	return tuple(res)


def convertAllTransformPipelines(policies: PoliciesT, paramsSpaces: ParamSpaceSpec) -> typing.Mapping[str, typing.Iterable[ChoicesT]]:
	return {k: convertSinglePolicy(v, paramsSpaces) for k, v in policies.items()}


def combinePolicies(transPols):
	combined = []
	for pol in transPols.values():
		combined.extend(pol)

	return combined


def aggregateOps(combined: typing.Iterable[ChoicesT]) -> typing.Mapping[str, typing.Iterable[ActionParamsT]]:
	piJoint = defaultdict(list)

	for transformPipeline in combined:
		for op_name, p, val in transformPipeline:
			piJoint[op_name].append((p, val))
	return piJoint


def getConvolvedDistribution(md: multivariate_normal_frozen, bounds: MultivariateBoundsT, axis: int, resolution: typing.Tuple[int, int] = (1000, 10000)) -> typing.Tuple[np.ndarray, np.ndarray, NumT]:
	if axis == 1:
		resolution = (resolution[1], resolution[0])

	xs, ys, dd = computeFunctionOnGrid(md.dist.pdf, (0, 1, resolution[0]), (*bounds[1], resolution[1]))
	axTicks = (xs, ys)

	dd = dd.T

	rawConvolved = (dd.T * xs).T

	xTicks = axTicks[axis]
	step = xTicks[1] - xTicks[0]

	if axis == 1:
		otherAxis = 0
	else:
		otherAxis = 1

	s = np.sum(rawConvolved, axis=otherAxis)
	s /= simpson(s, dx=step)  # normalization
	return xTicks, s, step


def plotFittedDistribution(x: np.ndarray, y: np.ndarray, fittedDist: scipy.stats._distn_infrastructure.rv_frozen, name: str) -> None:
	from matplotlib import pyplot as plt  # pylint:disable=import-outside-toplevel

	plt.plot(x, y, label="actual")
	plt.plot(x, fittedDist.pdf(x), label="norm")
	plt.legend()
	plt.title(name)
	plt.show()


def fitAndPlotConvolvedDist(dist: typing.Union[scipy.stats._continuous_distns.norm_gen, scipy.stats._continuous_distns.truncnorm_gen], md: multivariate_normal_frozen, bounds: MultivariateBoundsT, axis: int, name: str, method=differential_evolution) -> None:
	x, s, dx = getConvolvedDistribution(md, bounds, axis=axis)  # pylint:disable=unused-variable
	res = dist.fitPDFMLE(x, s, method=method)  # pylint:disable=unused-variable
	ic(res)
	dist, error = res

	print(dist.dist._parse_args(*dist.args, **dist.kwds))
	plotFittedDistribution(x, s, dist, name)
	print(calcRealMean(x, s))
	print(error)
	return dist


def fitAndPlot1DDist(points: np.ndarray, bounds: BoundsT, name: str, method=differential_evolution) -> None:
	dist, loss = scipy.stats.truncnorm.fitPointsMLE(points, bounds=bounds, method=method)
	print(dist.dist._parse_args(*dist.args, **dist.kwds), loss)
	binEdges, freqs = dist.adaptiveHistogram(points, density=True)

	curveX = np.linspace(binEdges[0], binEdges[-1], 1000)

	from matplotlib import pyplot as plt  # pylint:disable=import-outside-toplevel

	plt.plot(curveX, dist.pdf(curveX), label="fitted")
	plotPrecomputedHistogram(edges=binEdges, freqs=freqs, color="r", label="hist")
	plt.grid()
	plt.legend()
	plt.title(name)
	plt.show()
	return dist


def genJSONObjForForJoint(jointD: MultivariateGaussian) -> typing.Dict[str, typing.Union[typing.List[NumT], NumT]]:
	return {
		"m": jointD.mode.tolist(),
		# "s": jointD.stds.tolist(),
		"v": (jointD.stds ** 2).tolist(),
		"r": jointD.angles,
	}


def genJSONObjForForJointFromUnivariate(uniVariateD: scipy.stats._distn_infrastructure.rv_continuous_frozen, axis: int) -> typing.Dict[str, typing.Union[typing.List[NumT], int]]:
	params = uniVariateD.dist._parse_args(*uniVariateD.args, **uniVariateD.kwds)
	m = params[1]
	s = params[2]

	m = np.ones(2) * m
	s = np.ones(2) * s
	m[axis + 1 :] = m[:axis] = 0
	s[axis + 1 :] = s[:axis] = 0

	return {"m": m.tolist(), "v": (s ** 2).tolist(), "r": 0}


def getJSONObjForUnivariate(uniVariateD: scipy.stats._distn_infrastructure.rv_continuous_frozen) -> typing.List[typing.Union[str, NumT]]:
	params = uniVariateD.dist._parse_args(*uniVariateD.args, **uniVariateD.kwds)
	m = params[1]
	s = params[2]
	return ["norm", m, s ** 2]


defaultSpecBundleDir = repoRootDir / "PreImgAugment/autoaugment/AugmentersSpec/specBundle"


DistSpecT = typing.Dict[str, typing.Dict[str, typing.Any]]
Augs2ProcessParamT = typing.Optional[typing.List[str]]


class Pipeline:
	__slots__ = ("specBundleDir", "policies", "paramsSpaces", "transPols", "combinedPolicy", "opss", "distsFile", "dists")

	def __init__(self, specBundleDir: Path = defaultSpecBundleDir) -> None:
		self.specBundleDir = specBundleDir
		self.prepare()

	def prepare(self) -> None:
		self.policies = getPolicies()
		self.paramsSpaces = getParamsSpaces()
		self.transPols = convertAllTransformPipelines(self.policies, self.paramsSpaces)
		self.combinedPolicy = combinePolicies(self.transPols)
		self.opss = aggregateOps(self.combinedPolicy)

		self.distsFile = self.specBundleDir / "dists.json"
		if self.distsFile.is_file():
			self.dists = json.loads(self.distsFile.read_text())
		else:
			self.dists = {}

	def __call__(self, augmentersToProcess: Augs2ProcessParamT = None, method=differential_evolution) -> None:
		self.fitAndSaveDists(augmentersToProcess=augmentersToProcess, method=method)
		self.fitAndSaveMarkovChain()

	def fitAndSaveMarkovChain(self) -> None:
		markovChainFiles = MarkovChainParamsFiles(directory=self.specBundleDir)
		sp = AugmentersSpec.loadFromParamsDists(self.dists, None)
		mc = trainMarkovChain(sp, self.combinedPolicy)
		mc.save(markovChainFiles)

	def fitAndSaveDists(self, augmentersToProcess: Augs2ProcessParamT = None, method=differential_evolution) -> None:
		dists = self.fitDists(augmentersToProcess, method=method)
		if dists:
			self.saveDists()

	def saveDists(self) -> None:
		self.distsFile.write_text(postProcessSpecJSON(json.dumps(dict(sorted(self.dists.items(), key=lambda x: x[0])), indent="\t")))

	def fitDists(self, augmentersToProcess: Augs2ProcessParamT = None, method=differential_evolution) -> DistSpecT:
		opss = deepcopy(self.opss)
		detectAndInjectGroups(opss, self.paramsSpaces)

		from matplotlib import pyplot as plt  # pylint:disable=import-outside-toplevel
		from StatsUtils.plot import seabornJointPlotWithGaussian  # pylint:disable=import-outside-toplevel

		dists = {}

		if augmentersToProcess is not None:
			opss = {op: opss[op] for op in augmentersToProcess}

		for op, points in opss.items():
			isSigned = self.paramsSpaces[op][1]
			points = np.array(points)
			dists[op] = subRes = {}
			if points[0][1] is not None:
				md: StatsUtils.dists.MultivariateGaussian.MultivariateGaussian
				jp, md, bounds = seabornJointPlotWithGaussian(points, alpha=0.2)
				subRes["joint"] = genJSONObjForForJoint(md)
				jp.fig.suptitle(op)
				plt.show()
				bounds = md.computeBounds(0.001)
				probRes = fitAndPlotConvolvedDist(scipy.stats.truncnorm, md, bounds, axis=0, name="P", method=method)
				subRes["prob"] = probRes.mean()
				paramsRes = fitAndPlotConvolvedDist(scipy.stats.norm, md, bounds, axis=1, name="I", method=method)
				subRes["params"] = getJSONObjForUnivariate(paramsRes)

				if isSigned:
					negPoints = np.array(points)
					negPoints[:, 1] = -negPoints[:, 1]
					biSignedPoints = np.vstack([points, negPoints])
					del negPoints
					del points
					jp, md, bounds = seabornJointPlotWithGaussian(biSignedPoints, alpha=0.2)
					jp.fig.suptitle(op + "_bi-signed")
					plt.show()
					bounds = md.computeBounds(0.001)
					paramsRes = fitAndPlotConvolvedDist(scipy.stats.norm, md, bounds, axis=1, name="I", method=method)
					bothSignValDistDescriptor = getJSONObjForUnivariate(paramsRes)
					bothSignValDistDescriptor[1] = 0  # bisigned dist must have zero mean/mode
					subRes["bothSigns"] = bothSignValDistDescriptor
			else:
				points = points.T[0].astype(np.float)
				probRes = fitAndPlot1DDist(points, (0, 1), op + " (P)", method=method)
				subRes["joint"] = genJSONObjForForJointFromUnivariate(probRes, 0)
				subRes["prob"] = probRes.mean()
				subRes["params"] = None

		if dists:
			self.dists.update(dists)
		return dists


rxBank = [
	("\n\t+\\]", "]"),
	(": \\[\n", ": ["),
	("[\t\n]+(\\d+(\\.\\d+)?)", r"\1"),
	('\\[\t+("[^"]+")', r"[\1"),
	(",(\\d+)", r", \1"),
]
rxBank = [(re.compile(el[0]), el[1]) for el in rxBank]

groupRx = re.compile("^(.+)[XY]$")


def postProcessSpecJSON(res: str) -> str:
	for rx, sub in rxBank:
		res = rx.subn(sub, res)[0]
	return res


def detectAndInjectGroups(opss: typing.Mapping[str, typing.Iterable[ActionParamsT]], paramsSpaces: DistSpecT) -> None:
	groups = defaultdict(list)
	for el in opss.keys():
		m = groupRx.match(el)
		if m:
			gName = m.group(1)
			groups[gName].append(el)

	for groupName, ops in groups.items():
		if groupName in paramsSpaces or groupName in opss:
			raise KeyError("Op already exists", groupName)
		it = iter(ops)
		signedness = paramsSpaces[next(it)][1]
		for op in it:
			if paramsSpaces[op][1] != signedness:
				raise ValueError("Signedness is inconsistent within group", groupName)

		united = []
		for op in ops:
			united.extend(opss[op])
		opss[groupName] = united
		paramsSpaces[groupName] = (None, signedness)


def trainMarkovChain(sp: AugmentersSpec, policy: typing.Iterable[typing.Iterable[ActionParamsT, ...]]) -> MarkovChain:
	p0 = np.zeros(len(sp.name2idx))
	count1 = np.zeros(len(sp.name2idx))

	T = np.zeros((len(sp.name2idx), len(sp.name2idx)))

	for seq in policy:
		prevActIdx = 0

		isFirst = True
		for act in seq:
			name, prob, _intensity = act
			i = sp.name2idx[name]

			if isFirst:
				p0[i] += prob
				p0[0] += 1 - prob
				count1[i] += 1
				count1[0] += 1
				isFirst = False

			T[prevActIdx, i] += prob
			T[prevActIdx, 0] += 1 - prob
			prevActIdx = i

	count1[count1 == 0] = 1
	p0 = p0 / count1
	p0 /= np.sum(p0)

	norm = np.sum(T, axis=1)
	norm[norm == 0] = 1
	T = (T.T / norm).T

	return MarkovChain(p0=p0, T=T)
