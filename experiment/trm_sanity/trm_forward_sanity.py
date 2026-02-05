print("TRM sanity script started")

import sys
from pathlib import Path

# Add TinyRecursiveModels repo to PYTHONPATH
THIS_FILE = Path(__file__).resolve()

LMS_ROOT = THIS_FILE
while LMS_ROOT.name != "lms":
    LMS_ROOT = LMS_ROOT.parent

TRM_ROOT = LMS_ROOT / "repos" / "trm" / "TinyRecursiveModels"
sys.path.append(str(TRM_ROOT))



print("Added TRM root:", TRM_ROOT)

import torch
print("Torch OK:", torch.__version__)

from omegaconf import OmegaConf
print("OmegaConf OK")

from utils.functions import load_model_class

trm_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
print("TRM model import OK:", trm_cls.__name__)

