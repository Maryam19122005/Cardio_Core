"""
Mock Data Generator
Produces realistic ECG and heart sound (PCG) signals at 1000 SPS
Based on physiological models
"""

import math
import time
from cardiocore_constants import (
    SAMPLE_RATE, SAMPLES_PER_FRAME, Frame, Rhythm, Murmur
)

class MockDataGenerator:
    """Generates realistic cardiac signals for testing"""
    
    def __init__(self, bpm=72, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE):
        self.bpm = bpm
        self.rhythm = rhythm
        self.murmur = murmur
        self.start_time_ms = int(time.time() * 1000)
        self.sample_count = 0
        
    def get_heart_phase(self, sample_index):
        """Get phase in cardiac cycle (0.0 to 1.0)"""
        # Heartbeat period in samples
        beat_period = SAMPLE_RATE * 60 / self.bpm
        phase = (sample_index % beat_period) / beat_period
        return phase
    
    def generate_ecg_sample(self, sample_index):
        """
        Generate one ECG sample using PQRST model
        Normal ECG: P wave -> QRS complex -> T wave
        """
        phase = self.get_heart_phase(sample_index)
        
        # PQRST waveform (simplified but realistic)
        # P wave: 0.0 - 0.15
        # QRS complex: 0.15 - 0.30 (R peak at 0.20)
        # T wave: 0.30 - 0.70
        # Baseline: 0.70 - 1.0
        
        if phase < 0.10:
            # P wave (atrial depolarization)
            ecg = 40 * math.sin(math.pi * phase / 0.10)
        elif phase < 0.30:
            # QRS complex (ventricular depolarization)
            # R peak at phase 0.20
            qrs_phase = (phase - 0.15) / 0.15  # 0 to 1
            if qrs_phase < 0.5:
                # Q-R rising
                ecg = 200 * math.sin(math.pi * qrs_phase * 2)
            else:
                # R-S falling
                ecg = 200 * math.cos(math.pi * (qrs_phase - 0.5) * 2)
        elif phase < 0.70:
            # T wave (ventricular repolarization)
            t_phase = (phase - 0.30) / 0.40
            ecg = 100 * math.sin(math.pi * t_phase)
        else:
            # Baseline
            ecg = 10 * math.sin(2 * math.pi * phase)
        
        # Add tiny noise
        noise = 5 * math.sin(sample_index * 0.1)
        return int(ecg + noise)
    
    def generate_pcg_sample(self, sample_index):
        """
        Generate PCG (heart sound) signal
        S1: systolic ejection sound (happens at 0.0-0.15 of cardiac cycle)
        S2: diastolic closure sound (happens at 0.65-0.75)
        """
        phase = self.get_heart_phase(sample_index)
        
        # S1: loud, low frequency (early systole)
        if 0.0 <= phase < 0.15:
            # S1 sound envelope
            s1_envelope = math.sin(math.pi * phase / 0.15)
            # Frequency components of S1 (fundamental ~60-80 Hz)
            s1 = 80 * s1_envelope * math.sin(2 * math.pi * 70 * phase / (1 / self.bpm * 60))
        else:
            s1 = 0
        
        # S2: diastolic closure sound (during diastole)
        if 0.65 <= phase < 0.80:
            # S2 sound envelope (slightly less loud than S1)
            s2_envelope = math.sin(math.pi * (phase - 0.65) / 0.15)
            # Frequency ~120 Hz (higher pitch than S1)
            s2 = 60 * s2_envelope * math.sin(2 * math.pi * 120 * (phase - 0.65) / (1 / self.bpm * 60))
        else:
            s2 = 0
        
        # Murmur injection (if enabled)
        murmur_signal = 0
        if self.murmur == Murmur.SYSTOLIC and 0.15 <= phase < 0.65:
            # Systolic murmur during systole
            murmur_signal = 30 * math.sin(2 * math.pi * 300 * phase)
        elif self.murmur == Murmur.DIASTOLIC and (phase < 0.15 or phase > 0.65):
            # Diastolic murmur
            murmur_signal = 25 * math.sin(2 * math.pi * 250 * phase)
        
        # Apply rhythm variations
        if self.rhythm == Rhythm.BRADY:
            # Slower - already baked into self.bpm
            pass
        elif self.rhythm == Rhythm.TACHY:
            # Faster - already baked into self.bpm
            pass
        elif self.rhythm == Rhythm.AF:
            # Irregular rhythm - add timing jitter
            jitter = 5 * math.sin(sample_index * 0.05)
            s1 += jitter
            s2 += jitter
        
        pcg = s1 + s2 + murmur_signal
        
        # Add noise
        noise = 3 * math.sin(sample_index * 0.15)
        return int(pcg + noise)
    
    def generate_frame(self):
        """Generate one complete frame (10 samples at 1000 SPS = 10ms)"""
        timestamp_ms = self.start_time_ms + self.sample_count
        frame_number = self.sample_count // SAMPLES_PER_FRAME
        
        ecg_samples = []
        pcg_samples = []
        
        for i in range(SAMPLES_PER_FRAME):
            idx = self.sample_count + i
            ecg_samples.append(self.generate_ecg_sample(idx))
            pcg_samples.append(self.generate_pcg_sample(idx))
        
        self.sample_count += SAMPLES_PER_FRAME
        
        return Frame(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            ecg_samples=ecg_samples,
            pcg_samples=pcg_samples
        )

# ============ TEST ============
if __name__ == '__main__':
    print("Testing Mock Data Generator...")
    gen = MockDataGenerator(bpm=72, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE)
    
    for i in range(5):
        frame = gen.generate_frame()
        print(f"\nFrame {frame.frame_number}:")
        print(f"  Timestamp: {frame.timestamp_ms} ms")
        print(f"  ECG: {frame.ecg_samples}")
        print(f"  PCG: {frame.pcg_samples}")
    
    print("\n✓ Mock generator working!")
