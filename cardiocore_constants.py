"""
CardioCore Constants & Data Structures
Defines all shared data types and configuration
"""

import dataclasses
from typing import List
from enum import Enum

# ============ CONFIG ============
SAMPLE_RATE = 1000  # 1000 samples per second
SAMPLES_PER_FRAME = 10  # Send 10 samples per frame (10ms)
FRAME_INTERVAL_MS = 10  # milliseconds
BUFFER_SIZE = 5000  # Ring buffer stores last 5000 samples = 5 seconds

# Signal filtering bands
ECG_LOWCUT = 0.5  # Hz
ECG_HIGHCUT = 40   # Hz
ECG_NOTCH = 50     # Hz (power line)
PCG_LOWCUT = 20    # Hz
PCG_HIGHCUT = 600  # Hz

# Heart timing (milliseconds, typical values)
R_TO_S1_MIN = 50   # R peak to S1: 50-150ms
R_TO_S1_MAX = 150
S1_TO_S2_MIN = 350  # S1 to S2: 350-650ms (systole)
S1_TO_S2_MAX = 650

# ============ ENUMS ============
class Rhythm(Enum):
    NORMAL = "normal"
    BRADY = "brady"      # Slow
    TACHY = "tachy"      # Fast
    AF = "afib"          # Atrial fibrillation

class Murmur(Enum):
    NONE = "none"
    SYSTOLIC = "systolic"
    DIASTOLIC = "diastolic"

# ============ DATA CLASSES ============
@dataclasses.dataclass
class Frame:
    """One frame of sensor data from ESP32"""
    frame_number: int
    timestamp_ms: int
    ecg_samples: List[int]  # 10 samples
    pcg_samples: List[int]  # 10 samples

@dataclasses.dataclass
class DetectionEvent:
    """R-peak, S1, or S2 detection"""
    event_type: str  # "r_peak", "s1", "s2"
    sample_index: int
    timestamp_ms: int
    confidence: float  # 0.0 to 1.0

@dataclasses.dataclass
class TwinState:
    """The complete cardiac state sent to Unity"""
    timestamp_ms: int
    
    # Current sample values
    ecg: float
    pcg: float
    
    # Heart rate
    bpm: int
    
    # Last detected events
    last_r_peak_ms: float  # When last R-peak occurred
    last_s1_ms: float
    last_s2_ms: float
    
    # Phase in cardiac cycle (0.0 to 1.0)
    # 0.0 = R peak, 1.0 = end of diastole (before next R)
    cardiac_phase: float
    
    # Systole vs Diastole (0.0 to 1.0 within that phase)
    systole_phase: float  # 0.0-1.0 during systole (S1→S2), -1 if not in systole
    diastole_phase: float  # 0.0-1.0 during diastole (S2→S1), -1 if not in diastole
    
    # Classification
    rhythm: Rhythm
    murmur: Murmur
    
    # Signal quality
    lead_off: bool  # True if sensor disconnected
    ecg_confidence: float
    pcg_confidence: float

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            'timestamp_ms': self.timestamp_ms,
            'ecg': float(self.ecg),
            'pcg': float(self.pcg),
            'bpm': self.bpm,
            'last_r_peak_ms': self.last_r_peak_ms,
            'last_s1_ms': self.last_s1_ms,
            'last_s2_ms': self.last_s2_ms,
            'cardiac_phase': round(self.cardiac_phase, 3),
            'systole_phase': round(self.systole_phase, 3),
            'diastole_phase': round(self.diastole_phase, 3),
            'rhythm': self.rhythm.value,
            'murmur': self.murmur.value,
            'lead_off': self.lead_off,
            'ecg_confidence': round(self.ecg_confidence, 3),
            'pcg_confidence': round(self.pcg_confidence, 3),
        }
