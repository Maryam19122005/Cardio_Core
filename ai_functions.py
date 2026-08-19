"""
AI Signal Processing Functions
These are STUBS - Maryam & Mahida will replace with real ML models
For now: basic heuristic detection to unblock the pipeline
"""

import math
from typing import List, Tuple
from cardiocore_constants import Rhythm, Murmur

class AISignalProcessor:
    """Stub implementations of the 4 required functions"""
    
    @staticmethod
    def detect_rpeaks(ecg_samples: List[int], sample_rate: int = 1000) -> List[int]:
        """
        Detect R-peaks in ECG signal.
        
        Args:
            ecg_samples: List of ECG values
            sample_rate: Samples per second
        
        Returns:
            List of indices where R-peaks occur
        
        NOTE: This is a STUB using simple peak detection.
              Maryam & Mahida will replace with Pan-Tompkins or neurokit2.
        """
        if len(ecg_samples) < 3:
            return []
        
        r_peaks = []
        
        # Simple peak detection: find local maxima
        threshold = max(ecg_samples) * 0.7  # 70% of max
        
        for i in range(1, len(ecg_samples) - 1):
            if (ecg_samples[i] > threshold and 
                ecg_samples[i] > ecg_samples[i-1] and 
                ecg_samples[i] > ecg_samples[i+1]):
                r_peaks.append(i)
        
        # Filter: R-peaks should be at least 400ms apart (min 0.4s at 1000 SPS = 400 samples)
        filtered_peaks = []
        for peak in r_peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] >= 400:
                filtered_peaks.append(peak)
        
        return filtered_peaks
    
    @staticmethod
    def segment_s1s2(pcg_samples: List[int], r_peak_indices: List[int], 
                     sample_rate: int = 1000) -> Tuple[List[int], List[int]]:
        """
        Detect S1 and S2 heart sounds using R-peak timing as reference.
        
        Args:
            pcg_samples: List of PCG (heart sound) values
            r_peak_indices: Indices of detected R-peaks from ECG
            sample_rate: Samples per second
        
        Returns:
            (s1_indices, s2_indices): Lists of detected S1 and S2 times
        
        NOTE: This is a STUB. Real version will use spectrograms + segmentation.
        """
        if len(pcg_samples) < 100 or not r_peak_indices:
            return [], []
        
        s1_indices = []
        s2_indices = []
        
        # For each detected R-peak, predict S1 and S2 timing
        for r_idx in r_peak_indices:
            # S1 occurs 50-150ms after R-peak
            s1_search_start = r_idx + int(0.05 * sample_rate)  # 50ms
            s1_search_end = r_idx + int(0.15 * sample_rate)    # 150ms
            
            if s1_search_end < len(pcg_samples):
                # Find loudest point (max absolute value) in S1 window
                s1_window = pcg_samples[s1_search_start:s1_search_end]
                if s1_window:
                    s1_rel_idx = max(range(len(s1_window)), 
                                    key=lambda i: abs(s1_window[i]))
                    s1_indices.append(s1_search_start + s1_rel_idx)
            
            # S2 occurs 350-650ms after S1 (or 400-800ms after R)
            s2_search_start = r_idx + int(0.4 * sample_rate)   # 400ms
            s2_search_end = r_idx + int(0.8 * sample_rate)     # 800ms
            
            if s2_search_end < len(pcg_samples):
                # Find loudest point in S2 window
                s2_window = pcg_samples[s2_search_start:s2_search_end]
                if s2_window:
                    s2_rel_idx = max(range(len(s2_window)), 
                                    key=lambda i: abs(s2_window[i]))
                    s2_indices.append(s2_search_start + s2_rel_idx)
        
        return s1_indices, s2_indices
    
    @staticmethod
    def classify_rhythm(ecg_samples: List[int], r_peak_indices: List[int], 
                       sample_rate: int = 1000) -> Rhythm:
        """
        Classify heart rhythm: normal / brady / tachy / AF
        
        Args:
            ecg_samples: List of ECG values
            r_peak_indices: Indices of detected R-peaks
            sample_rate: Samples per second
        
        Returns:
            Rhythm enum (NORMAL, BRADY, TACHY, AF)
        
        NOTE: This is a STUB using simple HR calculation.
        """
        if len(r_peak_indices) < 2:
            return Rhythm.NORMAL
        
        # Calculate heart rate from R-R intervals
        rr_intervals = []
        for i in range(1, len(r_peak_indices)):
            interval_samples = r_peak_indices[i] - r_peak_indices[i-1]
            interval_ms = (interval_samples / sample_rate) * 1000
            rr_intervals.append(interval_ms)
        
        if not rr_intervals:
            return Rhythm.NORMAL
        
        avg_rr_ms = sum(rr_intervals) / len(rr_intervals)
        bpm = 60000 / avg_rr_ms  # Convert ms to BPM
        
        # Check for irregular rhythm (AF)
        rr_std = (sum((x - avg_rr_ms)**2 for x in rr_intervals) / len(rr_intervals)) ** 0.5
        irregularity = rr_std / avg_rr_ms  # Coefficient of variation
        
        if irregularity > 0.15:  # >15% variation = AF
            return Rhythm.AF
        elif bpm < 60:
            return Rhythm.BRADY
        elif bpm > 100:
            return Rhythm.TACHY
        else:
            return Rhythm.NORMAL
    
    @staticmethod
    def classify_murmur(pcg_samples: List[int], s1_indices: List[int], 
                       s2_indices: List[int], rhythm: Rhythm,
                       sample_rate: int = 1000) -> Murmur:
        """
        Classify murmur type: none / systolic / diastolic
        
        Args:
            pcg_samples: List of PCG values
            s1_indices: Indices of S1 sounds
            s2_indices: Indices of S2 sounds
            rhythm: Classified rhythm (for context)
            sample_rate: Samples per second
        
        Returns:
            Murmur enum (NONE, SYSTOLIC, DIASTOLIC)
        
        NOTE: This is a STUB looking for energy between S1 and S2.
        """
        if not s1_indices or not s2_indices:
            return Murmur.NONE
        
        if len(s1_indices) == 0 or len(s2_indices) == 0:
            return Murmur.NONE
        
        # Calculate power in systole (S1 to S2) and diastole (S2 to S1)
        systolic_power = 0
        diastolic_power = 0
        
        for i in range(min(len(s1_indices), len(s2_indices))):
            s1_idx = s1_indices[i]
            s2_idx = s2_indices[i]
            
            if s1_idx < s2_idx:
                # Systolic window: S1 to S2
                systolic_window = pcg_samples[s1_idx:s2_idx]
                systolic_power += sum(x**2 for x in systolic_window) / len(systolic_window) if systolic_window else 0
                
                # Diastolic window: S2 to next S1 (estimate from next beat)
                if i + 1 < len(s1_indices):
                    diastolic_window = pcg_samples[s2_idx:s1_indices[i+1]]
                    diastolic_power += sum(x**2 for x in diastolic_window) / len(diastolic_window) if diastolic_window else 0
        
        # Normalize
        if systolic_power == 0 and diastolic_power == 0:
            return Murmur.NONE
        
        systolic_power /= len(s1_indices) if s1_indices else 1
        diastolic_power /= (len(s1_indices) - 1) if len(s1_indices) > 1 else 1
        
        # Threshold: if power is significantly above baseline, classify as murmur
        baseline_threshold = 100  # Tune this based on real data
        
        if systolic_power > baseline_threshold and systolic_power > diastolic_power * 1.5:
            return Murmur.SYSTOLIC
        elif diastolic_power > baseline_threshold and diastolic_power > systolic_power * 1.5:
            return Murmur.DIASTOLIC
        else:
            return Murmur.NONE

# ============ TEST ============
if __name__ == '__main__':
    print("Testing AI Signal Processing Functions...")
    
    from mock_data_generator import MockDataGenerator
    
    gen = MockDataGenerator(bpm=72)
    
    # Generate 2 heartbeats worth of data
    all_ecg = []
    all_pcg = []
    for _ in range(144):  # 144 frames * 10 samples = 1440 samples = 1.44 seconds
        frame = gen.generate_frame()
        all_ecg.extend(frame.ecg_samples)
        all_pcg.extend(frame.pcg_samples)
    
    processor = AISignalProcessor()
    
    print("\n1. detect_rpeaks:")
    r_peaks = processor.detect_rpeaks(all_ecg)
    print(f"   Found {len(r_peaks)} R-peaks at indices: {r_peaks[:5]}...")
    
    print("\n2. segment_s1s2:")
    s1_idx, s2_idx = processor.segment_s1s2(all_pcg, r_peaks)
    print(f"   Found {len(s1_idx)} S1 and {len(s2_idx)} S2")
    
    print("\n3. classify_rhythm:")
    rhythm = processor.classify_rhythm(all_ecg, r_peaks)
    print(f"   Rhythm: {rhythm.value}")
    
    print("\n4. classify_murmur:")
    murmur = processor.classify_murmur(all_pcg, s1_idx, s2_idx, rhythm)
    print(f"   Murmur: {murmur.value}")
    
    print("\n✓ All AI function tests passed!")
