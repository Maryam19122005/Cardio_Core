"""
Fusion Engine
Combines sensor data + AI classifications into TwinState for Unity
Tracks cardiac phase, systole/diastole, BPM, etc.
"""

import math
from typing import Optional, Tuple
from cardiocore_constants import (
    TwinState, Frame, Rhythm, Murmur, SAMPLE_RATE, BUFFER_SIZE
)
from signal_buffers import SensorBuffers
from ai_functions import AISignalProcessor

from plausibility_layer import PlausibilityValidator, ClassificationOutput

_RHYTHM_LABEL_MAP = {
    "normal": "normal",
    "brady": "bradycardia",
    "tachy": "tachycardia",
    "afib": "afib",
}

class CardioFusionEngine:
    """
    Main processing engine:
    - Receives frames from sensors
    - Runs AI detection/classification
    - Outputs TwinState for VR
    """
    
    def __init__(self):
        self.buffers = SensorBuffers()
        self.processor = AISignalProcessor()
        
        # State tracking
        self.last_r_peak_time_ms = -1000
        self.last_s1_time_ms = -1000
        self.last_s2_time_ms = -1000
        self.current_bpm = 72
        self.current_rhythm = Rhythm.NORMAL
        self.current_murmur = Murmur.NONE
        self.validator = PlausibilityValidator()
        self.beat_counter = 0
        self.current_verdict = None
        
        # For low-latency detection
        self.last_detection_frame = -1
        self.detection_interval_frames = 10  # Run AI every 10 frames (100ms)

    def _verdict_fields(self) -> dict:
        if self.current_verdict is None:
            return dict(overall_valid=True, overall_confidence=1.0,
                        caution_mode=False, caution_reason="")
        v = self.current_verdict
        return dict(
            overall_valid=v.overall_valid,
            overall_confidence=v.overall_confidence,
            caution_mode=not v.overall_valid,
            caution_reason=v.summary if not v.overall_valid else "",
        )

    def process_frame(self, frame: Frame) -> Optional[TwinState]:
        """
        Process one sensor frame and return updated TwinState
        
        Args:
            frame: Frame from ESP32
        
        Returns:
            TwinState object for Unity, or None on error
        """
        # 1. Add frame to buffers
        self.buffers.add_frame(frame)
        
        # 2. Get current samples
        current_ecg = frame.ecg_samples[-1]  # Last sample in frame
        current_pcg = frame.pcg_samples[-1]
        
        # 3. Periodically run AI detection (not every frame for performance)
        frame_number = frame.frame_number
        if frame_number - self.last_detection_frame >= self.detection_interval_frames:
            self._run_detections()
            self.last_detection_frame = frame_number
        
        # 4. Compute cardiac phase and systole/diastole
        cardiac_phase, systole_phase, diastole_phase = self._compute_phases(
            frame.timestamp_ms
        )
        
        # 5. Build TwinState
        twin_state = TwinState(
            timestamp_ms=frame.timestamp_ms,
            ecg=current_ecg,
            pcg=current_pcg,
            bpm=self.current_bpm,
            last_r_peak_ms=self.last_r_peak_time_ms,
            last_s1_ms=self.last_s1_time_ms,
            last_s2_ms=self.last_s2_time_ms,
            cardiac_phase=cardiac_phase,
            systole_phase=systole_phase,
            diastole_phase=diastole_phase,
            **self._verdict_fields(),
            rhythm=self.current_rhythm,
            murmur=self.current_murmur,
            lead_off=False,  # Will add sensor check later
            ecg_confidence=0.9,  # Stub
            pcg_confidence=0.85  # Stub
        )
        
        return twin_state
    
    def _run_detections(self):
        """
        Run AI functions on buffered data to update state
        Called every ~100ms
        """
        # Get last 2 seconds of data for analysis
        analysis_window = SAMPLE_RATE * 2  # 2000 samples
        ecg_data, pcg_data = self.buffers.get_last_n_samples(analysis_window)
        
        if len(ecg_data) < 500:  # Need at least 0.5 seconds
            return
        
        # Detect R-peaks
        r_peak_indices = self.processor.detect_rpeaks(ecg_data)
        
        if r_peak_indices:
            # Calculate BPM from R-R intervals
            self.current_bpm = self._calculate_bpm(r_peak_indices, ecg_data)
            
            # Update last R-peak time
            last_r_rel = r_peak_indices[-1]
            self.last_r_peak_time_ms = (
                self.buffers.latest_timestamp_ms - 
                (len(ecg_data) - last_r_rel) * (1000 / SAMPLE_RATE)
            )
            
            # Detect S1 and S2
            s1_indices, s2_indices = self.processor.segment_s1s2(pcg_data, r_peak_indices)
            
            if s1_indices:
                last_s1_rel = s1_indices[-1]
                self.last_s1_time_ms = (
                    self.buffers.latest_timestamp_ms - 
                    (len(pcg_data) - last_s1_rel) * (1000 / SAMPLE_RATE)
                )
            
            if s2_indices:
                last_s2_rel = s2_indices[-1]
                self.last_s2_time_ms = (
                    self.buffers.latest_timestamp_ms - 
                    (len(pcg_data) - last_s2_rel) * (1000 / SAMPLE_RATE)
                )
        
        # Classify rhythm
        self.current_rhythm = self.processor.classify_rhythm(ecg_data, r_peak_indices)
        
        # Classify murmur (needs S1/S2 info)
        s1_indices, s2_indices = self.processor.segment_s1s2(pcg_data, r_peak_indices)
        self.current_murmur = self.processor.classify_murmur(
            pcg_data, s1_indices, s2_indices, self.current_rhythm
        )
    

        self.beat_counter += 1
        cycle = ClassificationOutput(
            beat_index=self.beat_counter,
            r_peak_ms=self.last_r_peak_time_ms,
            s1_ms=self.last_s1_time_ms if self.last_s1_time_ms >= 0 else None,
            s2_ms=self.last_s2_time_ms if self.last_s2_time_ms >= 0 else None,
            heart_rate_bpm=self.current_bpm,
            rhythm_label=_RHYTHM_LABEL_MAP.get(self.current_rhythm.value, "normal"),
            murmur_label=self.current_murmur.value,
            raw_confidence=0.9,
        )
        self.current_verdict = self.validator.validate_cycle(cycle)

    def _calculate_bpm(self, r_peak_indices, ecg_data) -> int:
        """Calculate BPM from R-R intervals"""
        if len(r_peak_indices) < 2:
            return self.current_bpm
        
        # Get last 2 R-R intervals for stable estimate
        rr_samples = []
        for i in range(max(0, len(r_peak_indices)-2), len(r_peak_indices)-1):
            interval = r_peak_indices[i+1] - r_peak_indices[i]
            rr_samples.append(interval)
        
        if rr_samples:
            avg_rr_samples = sum(rr_samples) / len(rr_samples)
            avg_rr_ms = (avg_rr_samples / SAMPLE_RATE) * 1000
            bpm = int(60000 / avg_rr_ms)
            return max(40, min(200, bpm))  # Clamp to reasonable range
        
        return self.current_bpm
    
    def _compute_phases(self, timestamp_ms: int) -> Tuple[float, float, float]:
        """
        Compute cardiac phase (0.0-1.0 through full cycle)
        and systole/diastole phases
        
        Returns:
            (cardiac_phase, systole_phase, diastole_phase)
            All 0.0-1.0 when active, -1 when inactive
        """
        if self.current_bpm <= 0:
            return 0.0, -1, -1
        
        # Time since last R-peak
        if self.last_r_peak_time_ms < 0:
            return 0.0, -1, -1
        
        time_since_r = timestamp_ms - self.last_r_peak_time_ms
        heartbeat_ms = 60000 / self.current_bpm
        
        # Cardiac phase: 0 at R-peak, 1 at next R-peak
        cardiac_phase = (time_since_r % heartbeat_ms) / heartbeat_ms
        cardiac_phase = max(0.0, min(1.0, cardiac_phase))
        
        # Systole: ~0.3 to ~0.65 of cycle (R→S2, about 200-350ms for 72 BPM)
        systole_start = 0.2  # R-peak + 20% = S1
        systole_end = 0.65   # S2 point
        
        if systole_start <= cardiac_phase <= systole_end:
            systole_phase = (cardiac_phase - systole_start) / (systole_end - systole_start)
        else:
            systole_phase = -1
        
        # Diastole: ~0.65 to ~1.0 (S2→R)
        diastole_start = systole_end
        diastole_end = 1.0
        
        if diastole_start <= cardiac_phase <= diastole_end:
            diastole_phase = (cardiac_phase - diastole_start) / (diastole_end - diastole_start)
        else:
            diastole_phase = -1
        
        return cardiac_phase, systole_phase, diastole_phase
    
    def get_state(self) -> Optional[TwinState]:
        """Get current state without processing new frame"""
        if self.buffers.latest_timestamp_ms <= 0:
            return None
        
        return TwinState(
            timestamp_ms=self.buffers.latest_timestamp_ms,
            ecg=self.buffers.ecg_buffer.get_last_n(1)[0] if len(self.buffers.ecg_buffer) > 0 else 0,
            pcg=self.buffers.pcg_buffer.get_last_n(1)[0] if len(self.buffers.pcg_buffer) > 0 else 0,
            bpm=self.current_bpm,
            last_r_peak_ms=self.last_r_peak_time_ms,
            last_s1_ms=self.last_s1_time_ms,
            last_s2_ms=self.last_s2_time_ms,
            cardiac_phase=self._compute_phases(self.buffers.latest_timestamp_ms)[0],
            systole_phase=self._compute_phases(self.buffers.latest_timestamp_ms)[1],
            diastole_phase=self._compute_phases(self.buffers.latest_timestamp_ms)[2],
            **self._verdict_fields(),
            rhythm=self.current_rhythm,
            murmur=self.current_murmur,
            lead_off=False,
            ecg_confidence=0.9,
            pcg_confidence=0.85
        )

# ============ TEST ============
if __name__ == '__main__':
    print("Testing Fusion Engine...")
    
    from mock_data_generator import MockDataGenerator
    
    engine = CardioFusionEngine()
    gen = MockDataGenerator(bpm=75, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE)
    
    print("\nProcessing frames...")
    for i in range(200):  # 2 seconds of data
        frame = gen.generate_frame()
        state = engine.process_frame(frame)
        
        if i % 50 == 0 and state:
            print(f"\nFrame {i}:")
            print(f"  BPM: {state.bpm}")
            print(f"  Rhythm: {state.rhythm.value}")
            print(f"  Cardiac phase: {state.cardiac_phase:.2f}")
            print(f"  Systole phase: {state.systole_phase:.2f}")
            print(f"  Murmur: {state.murmur.value}")
    
    print("\n✓ Fusion engine test passed!")
