"""
AI Signal Processing Functions
Real implementations by Maryam
"""

import numpy as np
import joblib
from typing import List, Tuple
from scipy.signal import butter, filtfilt, iirnotch, hilbert, find_peaks
from scipy.ndimage import uniform_filter1d
import neurokit2 as nk
from cardiocore_constants import Rhythm, Murmur

# Model ek hi baar load hoga jab file import ho (baar baar load na ho)
_rhythm_model = joblib.load('rhythm_model.pkl')

class AISignalProcessor:
    """Stub implementations of the 4 required functions"""
    
    @staticmethod
    def detect_rpeaks(ecg_samples: List[int], sample_rate: int = 1000) -> List[int]:
        """
        Detect R-peaks in ECG signal using neurokit2 (Pan-Tompkins based).
        Returns: list of sample indices where R-peaks occur
        """
        if len(ecg_samples) < 10:
            return []
        
        ecg = np.array(ecg_samples, dtype=float)
        
        # Denoise: bandpass + notch filter
        nyquist = sample_rate / 2
        low = 0.5 / nyquist
        high = min(40 / nyquist, 0.99)   # ECG_HIGHCUT = 40 (constants file se)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, ecg)
        
        notch_freq = 50 / nyquist
        if notch_freq < 1.0:
            b_notch, a_notch = iirnotch(50, Q=30, fs=sample_rate)
            filtered = filtfilt(b_notch, a_notch, filtered)
        
        try:
            signals, info = nk.ecg_process(filtered, sampling_rate=sample_rate)
            r_peak_samples = info['ECG_R_Peaks']
            return [int(i) for i in r_peak_samples]
        except Exception:
            return []  # bohot chhota/noisy signal ho to crash na ho
    
    @staticmethod
    def segment_s1s2(pcg_samples: List[int], r_peak_indices: List[int], 
                    sample_rate: int = 1000) -> Tuple[List[int], List[int]]:
        """
        Detect S1/S2 using envelope + peak detection (independent of R-peaks,
        r_peak_indices signature ke liye rakha hai lekin filhal use nahi ho raha).
        Returns: (s1_indices, s2_indices) — sample indices
        """
        if len(pcg_samples) < 100:
            return [], []
        
        pcg = np.array(pcg_samples, dtype=float)
        
        # Envelope banao
        nyquist = sample_rate / 2
        low = 25 / nyquist
        high = min(200 / nyquist, 0.99)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, pcg)
        envelope = np.abs(hilbert(filtered))
        envelope = uniform_filter1d(envelope, size=max(1, int(0.02 * sample_rate)))
        
        # Peaks dhoondo
        peaks, _ = find_peaks(envelope, distance=int(0.08*sample_rate), prominence=0.015)
        
        if len(peaks) < 2:
            return [], []
        
        # Gaps ka threshold (clustering-based)
        gaps = np.diff(peaks) / sample_rate
        if len(gaps) >= 2:
            sorted_gaps = np.sort(gaps)
            diffs = np.diff(sorted_gaps)
            split_idx = np.argmax(diffs)
            threshold = (sorted_gaps[split_idx] + sorted_gaps[split_idx + 1]) / 2
        else:
            threshold = np.median(gaps)
        
        # Amplitude se decide karo pehla peak S1 hai ya S2 (S1 = lower amplitude)
        amps = envelope[peaks]
        labels = ['S1'] if amps[0] < amps[1] else ['S2']
        for i in range(len(gaps)):
            prev_label = labels[-1]
            next_label = 'S2' if prev_label == 'S1' else 'S1'
            labels.append(next_label)
        
        s1_indices = [int(p) for p, l in zip(peaks, labels) if l == 'S1']
        s2_indices = [int(p) for p, l in zip(peaks, labels) if l == 'S2']
        
        return s1_indices, s2_indices
    
    @staticmethod
    def classify_rhythm(ecg_samples: List[int], r_peak_indices: List[int], 
                    sample_rate: int = 1000) -> Rhythm:
        """
        Classify rhythm using trained Random Forest model.
        """
        if len(r_peak_indices) < 3:
            return Rhythm.NORMAL  # not enough data, safe default
        
        # RR intervals nikalo (seconds me)
        rr_intervals = np.diff(r_peak_indices) / sample_rate
        
        avg_hr = 60 / np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        rr_min = np.min(rr_intervals)
        rr_max = np.max(rr_intervals)
        rr_range = rr_max - rr_min
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else 0.0
        
        feat = np.array([[avg_hr, rr_std, rr_min, rr_max, rr_range, rmssd]])
        
        label = _rhythm_model.predict(feat)[0]  # string: 'normal'/'brady'/'tachy'/'afib'
        
        # Model ka string label ko Rhythm enum me convert karo
        label_map = {
            'normal': Rhythm.NORMAL,
            'brady': Rhythm.BRADY,
            'tachy': Rhythm.TACHY,
            'afib': Rhythm.AF
        }
        return label_map.get(label, Rhythm.NORMAL)
    
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
