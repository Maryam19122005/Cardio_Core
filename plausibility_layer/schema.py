"""
schema.py
Defines the data format for communication between:
- Maryam's classification pipeline → This plausibility layer → Ramish's Unity
"""

from typing import Optional, List, Literal
from dataclasses import dataclass
from datetime import datetime

# ===== INPUT: What Maryam's functions output =====
@dataclass
class CardiacEvent:
    """A single detected cardiac event (R-peak, S1, S2, etc.)"""
    type: Literal["R_peak", "S1", "S2", "murmur_start", "murmur_end"]
    timestamp_ms: float  # Absolute time in milliseconds
    relative_to_previous_ms: Optional[float] = None  # Time since previous event
    amplitude: Optional[float] = None  # Signal amplitude if available
    frequency_hz: Optional[float] = None  # Dominant frequency if available
    
@dataclass
class HeartbeatCycle:
    """One complete cardiac cycle from one R-peak to the next"""
    r_peak_ms: float
    s1_ms: Optional[float] = None
    s2_ms: Optional[float] = None
    next_r_peak_ms: Optional[float] = None
    heart_rate_bpm: Optional[float] = None
    murmur_type: Optional[Literal["none", "systolic", "diastolic"]] = None
    
@dataclass
class ClassificationOutput:
    """What Maryam's pipeline produces for each beat"""
    beat_index: int
    r_peak_ms: float
    s1_ms: Optional[float]
    s2_ms: Optional[float]
    heart_rate_bpm: float
    rhythm_label: Literal["normal", "bradycardia", "tachycardia", "afib"]
    murmur_label: Literal["none", "systolic", "diastolic"]
    raw_confidence: float  # Maryam's own confidence (0-1)

# ===== OUTPUT: What this plausibility layer produces =====
@dataclass
class PlausibilityResult:
    """Validation result for a single cardiac event"""
    valid: bool  # Is this physiologically possible?
    confidence: float  # 0-1 confidence score
    reason: Optional[str] = None  # Why invalid (if applicable)
    warnings: List[str] = None  # Minor concerns that don't invalidate
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
            
@dataclass
class HeartbeatVerdict:
    """Complete validation for one heartbeat cycle"""
    beat_index: int
    r_peak_ms: float
    s1_result: PlausibilityResult
    s2_result: PlausibilityResult
    rhythm_result: PlausibilityResult
    murmur_result: PlausibilityResult
    overall_confidence: float  # Aggregated confidence
    overall_valid: bool
    summary: str  # Human-readable summary for demo
    
@dataclass
class UnityRenderCommand:
    """What Ramish's Unity should render"""
    # Confidence-driven rendering
    render_confidence: float  # 0-1, dim/fade 3D heart based on this
    show_rhythm: str  # "normal", "bradycardia", etc.
    show_murmur: str  # "none", "systolic", "diastolic"
    highlight_valves: List[str]  # Which valves to highlight
    play_haptic: bool  # Should haptic buzz trigger?
    caution_mode: bool  # True = show "signal uncertain" warning
    caution_reason: str  # Why caution mode is active
