"""
constants.py
All physiological constants from CardioCore MR Truth Sheet
Literature-backed values - DO NOT CHANGE without updating documentation
"""

# ===== ECG FILTERING (Section 1.1) =====
ECG_PASSBAND = (0.5, 150)  # Hz
ECG_NOTCH = 50  # Hz (power-line interference)

# ===== PCG FILTERING (Section 1.2) =====
PCG_PASSBAND = (20, 200)  # Hz (primary acoustic energy)
S1_FREQ_RANGE = (20, 100)  # Hz (S1 dominates lower end)
S2_FREQ_RANGE = (50, 200)  # Hz (S2 concentrates higher)

# ===== CARDIAC TIMINGS (Section 2) =====
# Based on normal resting HR of 60-75 BPM (~800ms cycle)
R_TO_S1_MIN = 30  # ms (electromechanical delay minimum)
R_TO_S1_MAX = 70  # ms (electromechanical delay maximum)
SYSTOLIC_MIN = 270  # ms (S1 to S2 minimum)
SYSTOLIC_MAX = 330  # ms (S1 to S2 maximum)
DIASTOLIC_MIN = 450  # ms (S2 to next S1 minimum)
DIASTOLIC_MAX = 470  # ms (S2 to next S1 maximum)

# ===== HEART RATE BOUNDARIES (for rhythm classification) =====
HR_BRADY_THRESHOLD = 60  # BPM (below this = bradycardia)
HR_TACHY_THRESHOLD = 100  # BPM (above this = tachycardia)
HR_NORMAL_MIN = 60
HR_NORMAL_MAX = 100

# ===== AUSCULTATION POINTS (Section 4) =====
# For Unity spatial mapping
AUSCULTATION_POINTS = {
    "aortic": {"ics": 2, "side": "right", "location": "parasternal"},
    "pulmonic": {"ics": 2, "side": "left", "location": "parasternal"},
    "tricuspid": {"ics": 4, "side": "left", "location": "lower sternal border"},
    "mitral": {"ics": 5, "side": "left", "location": "midclavicular line"}
}

# ===== CONFIDENCE SCORE WEIGHTS =====
WEIGHT_TIMING = 0.6  # Timing validity is most important
WEIGHT_SEQUENCE = 0.2  # Correct S1→S2→S1 sequence
WEIGHT_CONSISTENCY = 0.2  # Consistency across beats


def get_hr_adjusted_timings(heart_rate_bpm):
    """
    Adjust timing windows based on actual heart rate.
    Reference: Section 2 - timings vary with heart rate.
    """
    if heart_rate_bpm < 1:
        heart_rate_bpm = 60  # fallback
    
    # Scale factor: faster HR = shorter intervals
    # At 60 BPM, cycle = 1000ms; at 120 BPM, cycle = 500ms
    cycle_length_ms = 60000 / heart_rate_bpm
    reference_cycle = 800  # ms at 75 BPM
    
    scale = cycle_length_ms / reference_cycle
    
    return {
        "r_to_s1_min": int(R_TO_S1_MIN * scale),
        "r_to_s1_max": int(R_TO_S1_MAX * scale),
        "systolic_min": int(SYSTOLIC_MIN * scale),
        "systolic_max": int(SYSTOLIC_MAX * scale),
        "diastolic_min": int(DIASTOLIC_MIN * scale),
        "diastolic_max": int(DIASTOLIC_MAX * scale),
    }
