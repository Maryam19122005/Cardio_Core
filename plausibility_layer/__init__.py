"""
Physiological Plausibility Layer for CardioCore MR
"""

from .plausibility import PlausibilityValidator
from .schema import (
    CardiacEvent, HeartbeatCycle, ClassificationOutput,
    PlausibilityResult, HeartbeatVerdict, UnityRenderCommand
)
from .constants import *

__version__ = "1.0.0"
