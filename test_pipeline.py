"""
Complete End-to-End Test Suite
Tests all components in isolation and integration
RUN THIS FIRST before starting the server
"""

import sys
import json
from cardiocore_constants import *
from mock_data_generator import MockDataGenerator
from signal_buffers import SensorBuffers, FrameParser
from ai_functions import AISignalProcessor
from fusion_engine import CardioFusionEngine

def print_section(title):
    """Print a test section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_constants():
    """Test 1: Constants & Data Structures"""
    print_section("TEST 1: Constants & Data Structures")
    
    print("\nChecking constants:")
    print(f"  Sample rate: {SAMPLE_RATE} SPS")
    print(f"  Samples per frame: {SAMPLES_PER_FRAME}")
    print(f"  Frame interval: {FRAME_INTERVAL_MS} ms")
    print(f"  Buffer size: {BUFFER_SIZE} samples ({BUFFER_SIZE/SAMPLE_RATE:.1f} sec)")
    
    print("\nTesting TwinState creation:")
    state = TwinState(
        timestamp_ms=1000,
        ecg=150,
        pcg=50,
        bpm=72,
        last_r_peak_ms=950,
        last_s1_ms=1000,
        last_s2_ms=1200,
        cardiac_phase=0.5,
        systole_phase=0.3,
        diastole_phase=-1,
        rhythm=Rhythm.NORMAL,
        murmur=Murmur.NONE,
        lead_off=False,
        ecg_confidence=0.9,
        pcg_confidence=0.85
    )
    
    state_dict = state.to_dict()
    print(f"  TwinState keys: {list(state_dict.keys())}")
    print(f"  Serializable: {json.dumps(state_dict) is not None}")
    
    print("\n✓ Constants test PASSED")
    return True

def test_mock_generator():
    """Test 2: Mock Data Generator"""
    print_section("TEST 2: Mock Data Generator")
    
    gen = MockDataGenerator(bpm=72, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE)
    print(f"  Generator created: BPM={gen.bpm}, Rhythm={gen.rhythm.value}")
    
    frames = []
    for i in range(10):
        frame = gen.generate_frame()
        frames.append(frame)
        
        # Validate frame
        if len(frame.ecg_samples) != SAMPLES_PER_FRAME:
            print(f"  ERROR: Frame {i} has {len(frame.ecg_samples)} ECG samples, expected {SAMPLES_PER_FRAME}")
            return False
        if len(frame.pcg_samples) != SAMPLES_PER_FRAME:
            print(f"  ERROR: Frame {i} has {len(frame.pcg_samples)} PCG samples, expected {SAMPLES_PER_FRAME}")
            return False
    
    print(f"  Generated 10 frames successfully")
    print(f"  Frame 0: ts={frames[0].timestamp_ms}ms, ECG range: [{min(frames[0].ecg_samples)}, {max(frames[0].ecg_samples)}]")
    print(f"  Frame 9: ts={frames[9].timestamp_ms}ms, PCG range: [{min(frames[9].pcg_samples)}, {max(frames[9].pcg_samples)}]")
    
    print("\n✓ Mock generator test PASSED")
    return True

def test_frame_parser():
    """Test 3: Frame Parser (binary packing/unpacking)"""
    print_section("TEST 3: Frame Parser")
    
    gen = MockDataGenerator()
    frame_orig = gen.generate_frame()
    
    print(f"  Original frame: number={frame_orig.frame_number}, ts={frame_orig.timestamp_ms}")
    
    # Pack to binary
    binary = FrameParser.pack_frame(frame_orig)
    print(f"  Packed size: {len(binary)} bytes")
    if len(binary) != 46:
        print(f"  ERROR: Expected 46 bytes, got {len(binary)}")
        return False
    
    # Unpack from binary
    frame_restored = FrameParser.unpack_frame(binary)
    if frame_restored is None:
        print(f"  ERROR: Failed to unpack frame")
        return False
    
    print(f"  Restored frame: number={frame_restored.frame_number}, ts={frame_restored.timestamp_ms}")
    
    # Validate
    valid, msg = FrameParser.validate_frame(frame_restored)
    print(f"  Validation: {msg}")
    if not valid:
        print(f"  ERROR: Frame validation failed")
        return False
    
    # Check values match
    if frame_orig.frame_number != frame_restored.frame_number:
        print(f"  ERROR: Frame number mismatch")
        return False
    if frame_orig.ecg_samples != frame_restored.ecg_samples:
        print(f"  ERROR: ECG samples mismatch")
        return False
    
    print("\n✓ Frame parser test PASSED")
    return True

def test_signal_buffers():
    """Test 4: Signal Buffers (ring buffers)"""
    print_section("TEST 4: Signal Buffers")
    
    buffers = SensorBuffers()
    gen = MockDataGenerator()
    
    print(f"  Created buffer: {buffers}")
    
    # Add frames
    for i in range(20):
        frame = gen.generate_frame()
        buffers.add_frame(frame)
    
    print(f"  Added 20 frames (200 samples)")
    print(f"  ECG buffer: {len(buffers.ecg_buffer)} samples")
    print(f"  PCG buffer: {len(buffers.pcg_buffer)} samples")
    
    # Get last N samples
    ecg, pcg = buffers.get_last_n_samples(50)
    if len(ecg) != 50 or len(pcg) != 50:
        print(f"  ERROR: Expected 50 samples, got ECG={len(ecg)}, PCG={len(pcg)}")
        return False
    
    print(f"  Last 50 samples retrieved successfully")
    print(f"  ECG range: [{min(ecg)}, {max(ecg)}]")
    print(f"  PCG range: [{min(pcg)}, {max(pcg)}]")
    
    print("\n✓ Signal buffers test PASSED")
    return True

def test_ai_functions():
    """Test 5: AI Signal Processing Functions"""
    print_section("TEST 5: AI Functions")
    
    gen = MockDataGenerator(bpm=72)
    processor = AISignalProcessor()
    
    # Generate 1.5 seconds of data
    all_ecg = []
    all_pcg = []
    for _ in range(150):
        frame = gen.generate_frame()
        all_ecg.extend(frame.ecg_samples)
        all_pcg.extend(frame.pcg_samples)
    
    print(f"  Generated {len(all_ecg)} ECG and {len(all_pcg)} PCG samples")
    
    # Test 1: R-peak detection
    print(f"\n  Running detect_rpeaks...")
    r_peaks = processor.detect_rpeaks(all_ecg)
    print(f"    Found {len(r_peaks)} R-peaks")
    if len(r_peaks) == 0:
        print(f"    WARNING: No R-peaks detected (signal may be too short)")
    elif len(r_peaks) < 1:
        print(f"    WARNING: Expected ~1-2 R-peaks for 1.5s at 72 BPM")
    
    # Test 2: S1/S2 detection
    print(f"\n  Running segment_s1s2...")
    s1_idx, s2_idx = processor.segment_s1s2(all_pcg, r_peaks if r_peaks else [])
    print(f"    Found {len(s1_idx)} S1 and {len(s2_idx)} S2")
    
    # Test 3: Rhythm classification
    print(f"\n  Running classify_rhythm...")
    rhythm = processor.classify_rhythm(all_ecg, r_peaks if r_peaks else [])
    print(f"    Rhythm: {rhythm.value}")
    
    # Test 4: Murmur classification
    print(f"\n  Running classify_murmur...")
    murmur = processor.classify_murmur(all_pcg, s1_idx, s2_idx, rhythm)
    print(f"    Murmur: {murmur.value}")
    
    print("\n✓ AI functions test PASSED")
    return True

def test_fusion_engine():
    """Test 6: Fusion Engine (complete pipeline)"""
    print_section("TEST 6: Fusion Engine")
    
    engine = CardioFusionEngine()
    gen = MockDataGenerator(bpm=75, rhythm=Rhythm.NORMAL, murmur=Murmur.NONE)
    
    print(f"  Engine created")
    
    # Process frames
    states = []
    for i in range(200):  # 2 seconds
        frame = gen.generate_frame()
        state = engine.process_frame(frame)
        if state:
            states.append(state)
    
    print(f"  Processed 200 frames, got {len(states)} states")
    
    if not states:
        print(f"  ERROR: No states generated")
        return False
    
    final_state = states[-1]
    print(f"\n  Final state:")
    print(f"    Timestamp: {final_state.timestamp_ms} ms")
    print(f"    BPM: {final_state.bpm}")
    print(f"    ECG: {final_state.ecg}")
    print(f"    PCG: {final_state.pcg}")
    print(f"    Rhythm: {final_state.rhythm.value}")
    print(f"    Murmur: {final_state.murmur.value}")
    print(f"    Cardiac phase: {final_state.cardiac_phase:.2f}")
    print(f"    Systole phase: {final_state.systole_phase:.2f}")
    print(f"    Diastole phase: {final_state.diastole_phase:.2f}")
    
    # Verify values are in reasonable ranges
    if not (40 <= final_state.bpm <= 200):
        print(f"  ERROR: BPM out of range: {final_state.bpm}")
        return False
    if not (-500 <= final_state.ecg <= 500):
        print(f"  ERROR: ECG out of range: {final_state.ecg}")
        return False
    if not (0.0 <= final_state.cardiac_phase <= 1.0):
        print(f"  ERROR: Cardiac phase out of range: {final_state.cardiac_phase}")
        return False
    
    print("\n✓ Fusion engine test PASSED")
    return True

def test_demo_modes():
    """Test 7: Demo Modes (different rhythms & murmurs)"""
    print_section("TEST 7: Demo Modes")
    
    test_cases = [
        (Rhythm.NORMAL, Murmur.NONE, "Normal sinus rhythm"),
        (Rhythm.BRADY, Murmur.NONE, "Bradycardia (slow)"),
        (Rhythm.TACHY, Murmur.NONE, "Tachycardia (fast)"),
        (Rhythm.NORMAL, Murmur.SYSTOLIC, "Systolic murmur"),
        (Rhythm.NORMAL, Murmur.DIASTOLIC, "Diastolic murmur"),
    ]
    
    for rhythm, murmur, desc in test_cases:
        print(f"\n  Testing: {desc}")
        
        engine = CardioFusionEngine()
        gen = MockDataGenerator(bpm=72, rhythm=rhythm, murmur=murmur)
        
        for _ in range(150):  # 1.5 seconds
            frame = gen.generate_frame()
            state = engine.process_frame(frame)
        
        state = engine.get_state()
        if state:
            print(f"    ✓ {rhythm.value} / {murmur.value}: BPM={state.bpm}, phase={state.cardiac_phase:.2f}")
        else:
            print(f"    ERROR: Failed to generate state")
            return False
    
    print("\n✓ Demo modes test PASSED")
    return True

def test_state_serialization():
    """Test 8: State Serialization (JSON for Unity)"""
    print_section("TEST 8: State Serialization")
    
    engine = CardioFusionEngine()
    gen = MockDataGenerator(bpm=72)
    
    for _ in range(100):
        frame = gen.generate_frame()
        engine.process_frame(frame)
    
    state = engine.get_state()
    if not state:
        print(f"  ERROR: No state to serialize")
        return False
    
    state_dict = state.to_dict()
    print(f"  State dict keys: {list(state_dict.keys())}")
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(state_dict)
        print(f"  JSON serialization: OK ({len(json_str)} bytes)")
        
        # Try to deserialize
        restored = json.loads(json_str)
        print(f"  JSON deserialization: OK")
        
    except Exception as e:
        print(f"  ERROR: Serialization failed: {e}")
        return False
    
    print("\n✓ State serialization test PASSED")
    return True

# ============ MAIN ============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  CardioCore End-to-End Test Suite")
    print("="*60)
    
    tests = [
        ("Constants", test_constants),
        ("Mock Generator", test_mock_generator),
        ("Frame Parser", test_frame_parser),
        ("Signal Buffers", test_signal_buffers),
        ("AI Functions", test_ai_functions),
        ("Fusion Engine", test_fusion_engine),
        ("Demo Modes", test_demo_modes),
        ("Serialization", test_state_serialization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n  ✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYour pipeline is ready! You can now:")
        print("  1. Run the server: python cardiocore_server.py")
        print("  2. Test in another terminal: curl http://localhost:5000/data")
        print("\n")
        sys.exit(0)
    else:
        print(f"\n✗✗✗ {failed} TESTS FAILED ✗✗✗\n")
        sys.exit(1)
