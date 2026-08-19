"""
plausibility.py
Physiological Plausibility Layer for CardioCore MR
Validates cardiac events against Truth Sheet literature
"""

from typing import Optional, List, Dict, Tuple
import math
from constants import *
from schema import *

class PlausibilityValidator:
    """
    Validates cardiac events against physiological bounds.
    Acts as the "sanity check" between Maryam's AI and Ramish's Unity.
    """
    
    def __init__(self):
        self.beat_history: List[HeartbeatCycle] = []
        self.prev_beat_valid = True
        
    # ===== CORE VALIDATION METHODS =====
    
    def validate_cycle(self, cycle: ClassificationOutput) -> HeartbeatVerdict:
        """
        Validate one complete cardiac cycle.
        This is the main method Maryam's pipeline will call.
        """
        # Calculate adjusted timings based on actual HR
        timings = get_hr_adjusted_timings(cycle.heart_rate_bpm)
        
        # 1. Validate R→S1 timing (electromechanical delay)
        s1_result = self._validate_r_to_s1(
            r_peak=cycle.r_peak_ms,
            s1=cycle.s1_ms,
            min_delay=timings["r_to_s1_min"],
            max_delay=timings["r_to_s1_max"]
        )
        
        # 2. Validate S1→S2 timing (systolic duration)
        s2_result = self._validate_s1_to_s2(
            s1=cycle.s1_ms,
            s2=cycle.s2_ms,
            min_duration=timings["systolic_min"],
            max_duration=timings["systolic_max"]
        )
        
        # 3. Validate rhythm classification
        rhythm_result = self._validate_rhythm(
            hr=cycle.heart_rate_bpm,
            label=cycle.rhythm_label
        )
        
        # 4. Validate murmur classification
        murmur_result = self._validate_murmur(
            s1=cycle.s1_ms,
            s2=cycle.s2_ms,
            murmur_label=cycle.murmur_label
        )
        
        # 5. Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence([
            s1_result.confidence,
            s2_result.confidence,
            rhythm_result.confidence,
            murmur_result.confidence
        ])
        
        overall_valid = all([
            s1_result.valid,
            s2_result.valid,
            rhythm_result.valid,
            murmur_result.valid
        ])
        
        # 6. Generate summary
        summary = self._generate_summary(
            s1_result, s2_result, rhythm_result, murmur_result,
            overall_valid, overall_confidence
        )
        
        return HeartbeatVerdict(
            beat_index=cycle.beat_index,
            r_peak_ms=cycle.r_peak_ms,
            s1_result=s1_result,
            s2_result=s2_result,
            rhythm_result=rhythm_result,
            murmur_result=murmur_result,
            overall_confidence=overall_confidence,
            overall_valid=overall_valid,
            summary=summary
        )
    
    def get_unity_command(self, verdict: HeartbeatVerdict) -> UnityRenderCommand:
        """
        Convert validation verdict to Unity render command.
        This is what Ramish will use to control the 3D heart.
        """
        # Determine caution mode based on confidence
        caution_mode = verdict.overall_confidence < 0.7
        caution_reason = ""
        
        if caution_mode:
            reasons = []
            if not verdict.s1_result.valid:
                reasons.append(f"S1 timing invalid: {verdict.s1_result.reason}")
            if not verdict.s2_result.valid:
                reasons.append(f"S2 timing invalid: {verdict.s2_result.reason}")
            if not verdict.rhythm_result.valid:
                reasons.append(f"Rhythm invalid: {verdict.rhythm_result.reason}")
            if not verdict.murmur_result.valid:
                reasons.append(f"Murmur invalid: {verdict.murmur_result.reason}")
            caution_reason = "; ".join(reasons) if reasons else "Low confidence"
        
        # Determine valve highlighting based on murmur type
        highlight_valves = []
        murmur_label = verdict.murmur_result.reason if verdict.murmur_result.valid else "none"
        # Extract murmur type from reason (e.g., "Systolic murmur detected" → "systolic")
        if "systolic" in str(murmur_label).lower():
            # Systolic murmurs → Aortic/Pulmonic valves
            highlight_valves = ["aortic", "pulmonic"]
        elif "diastolic" in str(murmur_label).lower():
            # Diastolic murmurs → Mitral/Tricuspid valves
            highlight_valves = ["mitral", "tricuspid"]
        
        return UnityRenderCommand(
            render_confidence=verdict.overall_confidence,
            show_rhythm=verdict.rhythm_result.reason if verdict.rhythm_result.valid else "unknown",
            show_murmur=murmur_label,
            highlight_valves=highlight_valves,
            play_haptic=verdict.overall_valid and verdict.overall_confidence > 0.8,
            caution_mode=caution_mode,
            caution_reason=caution_reason
        )
    
    # ===== INDIVIDUAL VALIDATION FUNCTIONS =====
    
    def _validate_r_to_s1(self, r_peak: float, s1: Optional[float], 
                          min_delay: int, max_delay: int) -> PlausibilityResult:
        """Validate electromechanical delay (R→S1)"""
        if s1 is None:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason="S1 not detected"
            )
        
        delay = s1 - r_peak
        
        if delay < min_delay:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"R→S1 delay too short: {delay:.0f}ms (min {min_delay}ms)"
            )
        elif delay > max_delay:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"R→S1 delay too long: {delay:.0f}ms (max {max_delay}ms)"
            )
        else:
            # Confidence scales linearly between min and max
            # 1.0 at ideal (50ms), lower at edges
            ideal = (min_delay + max_delay) / 2
            range_width = max_delay - min_delay
            deviation = abs(delay - ideal) / (range_width / 2)
            confidence = max(0.7, 1.0 - deviation * 0.3)  # Never below 0.7
            
            return PlausibilityResult(
                valid=True,
                confidence=confidence,
                reason=f"R→S1 delay {delay:.0f}ms (normal range {min_delay}-{max_delay}ms)"
            )
    
    def _validate_s1_to_s2(self, s1: Optional[float], s2: Optional[float],
                           min_duration: int, max_duration: int) -> PlausibilityResult:
        """Validate systolic duration (S1→S2)"""
        if s1 is None or s2 is None:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason="S1 or S2 not detected"
            )
        
        duration = s2 - s1
        
        if duration < min_duration:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Systolic duration too short: {duration:.0f}ms (min {min_duration}ms)"
            )
        elif duration > max_duration:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Systolic duration too long: {duration:.0f}ms (max {max_duration}ms)"
            )
        else:
            # Similar confidence scaling
            ideal = (min_duration + max_duration) / 2
            range_width = max_duration - min_duration
            deviation = abs(duration - ideal) / (range_width / 2)
            confidence = max(0.7, 1.0 - deviation * 0.3)
            
            return PlausibilityResult(
                valid=True,
                confidence=confidence,
                reason=f"Systolic duration {duration:.0f}ms (normal range {min_duration}-{max_duration}ms)"
            )
    
    def _validate_rhythm(self, hr: float, label: str) -> PlausibilityResult:
        """Validate rhythm classification against heart rate"""
        # First, check if HR is in a physiological range
        if hr < 20:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Impossible heart rate: {hr:.1f} BPM (< 20 BPM)"
            )
        elif hr > 250:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Impossible heart rate: {hr:.1f} BPM (> 250 BPM)"
            )
        
        # Check if label matches HR
        if label == "normal" and (hr < HR_NORMAL_MIN or hr > HR_NORMAL_MAX):
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Labeled 'normal' but HR is {hr:.1f} BPM"
            )
        elif label == "bradycardia" and hr >= HR_BRADY_THRESHOLD:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Labeled 'bradycardia' but HR is {hr:.1f} BPM (>= {HR_BRADY_THRESHOLD} BPM)"
            )
        elif label == "tachycardia" and hr <= HR_TACHY_THRESHOLD:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Labeled 'tachycardia' but HR is {hr:.1f} BPM (<= {HR_TACHY_THRESHOLD} BPM)"
            )
        elif label == "afib":
            # AF detection is harder - just warn, don't invalidate
            return PlausibilityResult(
                valid=True,
                confidence=0.6,
                reason="AF detected (requires additional validation)",
                warnings=["AF classification should be verified with multiple beats"]
            )
        
        # Valid rhythm
        confidence = 0.9 if label == "normal" else 0.85
        return PlausibilityResult(
            valid=True,
            confidence=confidence,
            reason=f"Rhythm {label} at {hr:.1f} BPM"
        )
    
    def _validate_murmur(self, s1: Optional[float], s2: Optional[float],
                         murmur_label: str) -> PlausibilityResult:
        """
        Validate murmur classification against timing windows.
        Reference: Section 3 - Murmur Windows
        """
        if murmur_label == "none":
            return PlausibilityResult(
                valid=True,
                confidence=1.0,
                reason="No murmur detected"
            )
        
        if s1 is None or s2 is None:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason="Cannot validate murmur without S1/S2 timing"
            )
        
        # Murmur must occur in the correct window
        # Systolic: between S1 and S2
        # Diastolic: after S2, before next S1
        
        if murmur_label == "systolic":
            # Check if there's a reasonable time window between S1 and S2
            sys_duration = s2 - s1
            if sys_duration < 50:  # Too short for a murmur
                return PlausibilityResult(
                    valid=False,
                    confidence=0.0,
                    reason="Systolic murmur claimed but systole too short",
                    warnings=["Check for S1/S2 misdetection"]
                )
            return PlausibilityResult(
                valid=True,
                confidence=0.8,
                reason="Systolic murmur (between S1 and S2)"
            )
        
        elif murmur_label == "diastolic":
            # For diastolic, we need the next S1 to verify
            # Since we don't have it yet, we can only warn
            return PlausibilityResult(
                valid=True,
                confidence=0.6,
                reason="Diastolic murmur (requires next beat for validation)",
                warnings=["Diastolic murmur confirmed with next S1 timing"]
            )
        
        else:
            return PlausibilityResult(
                valid=False,
                confidence=0.0,
                reason=f"Unknown murmur type: {murmur_label}"
            )
    
    # ===== HELPER METHODS =====
    
    def _calculate_overall_confidence(self, confidences: List[float]) -> float:
        """Aggregate individual confidences into overall score"""
        if not confidences:
            return 0.0
        
        # Weighted average, with high weight on the lowest confidence
        # This is stricter than simple average
        min_conf = min(confidences)
        avg_conf = sum(confidences) / len(confidences)
        
        # If any confidence is 0, overall is 0
        if min_conf == 0:
            return 0.0
        
        # Blend: 70% weight to the lowest confidence, 30% to average
        return 0.7 * min_conf + 0.3 * avg_conf
    
    def _generate_summary(self, s1_result, s2_result, rhythm_result, 
                          murmur_result, overall_valid, overall_confidence) -> str:
        """Generate human-readable summary"""
        status = "✅ Valid" if overall_valid else "❌ Invalid"
        
        # Build quick diagnosis
        issues = []
        if not s1_result.valid:
            issues.append("S1 timing")
        if not s2_result.valid:
            issues.append("S2 timing")
        if not rhythm_result.valid:
            issues.append("rhythm")
        if not murmur_result.valid:
            issues.append("murmur")
        
        if not issues:
            summary = f"{status} All physiological parameters normal. Confidence: {overall_confidence:.2f}"
        else:
            summary = f"{status} Issues detected in: {', '.join(issues)}. Confidence: {overall_confidence:.2f}"
        
        return summary
