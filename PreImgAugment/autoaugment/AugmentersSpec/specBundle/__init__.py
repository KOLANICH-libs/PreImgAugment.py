from pathlib import Path

from .. import AugmentersSpec

thisDir = Path(__file__).absolute().parent
specBundleDir = thisDir
augSpec = AugmentersSpec.loadFromDir(specBundleDir)
