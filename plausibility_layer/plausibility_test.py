"""
plausibility_test.py
Test harness for the Physiological Plausibility Layer
Demonstrates the module working with sample data
"""

from plausibility import PlausibilityValidator
from schema import ClassificationOutput

def create_test_beat(beat_index, r_peak, s1, s2, hr, rhythm, murmur):
    """Helper to create test ClassificationOutput objects"""
    return ClassificationOutput(
        beat_index=beat_index,
        r_peak_ms=r_peak,
        s1_ms=s1,
        s2_ms=s2,
        heart_rate_bpm=hr,
        rhythm_label=rhythm,
        murmur_label=murmur,
        raw_confidence=0.9  # Maryam's confidence (not used by our layer)
    )

def run_tests():
    """Run through various test scenarios"""
    validator = PlausibilityValidator()
    
    print("=" * 60)
    print("PHYSIOLOGICAL PLAUSIBILITY LAYER - TEST HARNESS")
    print("=" * 60)
    
    # Test 1: Normal healthy beat
    print("\n[TEST 1] Normal heartbeat (60 BPM)")
    normal_beat = create_test_beat(
        beat_index=1,
        r_peak=100,
        s1=150,  # 50ms after R-peak (normal)
        s2=430,  # 280ms after S1 (normal)
        hr=72,
        rhythm="normal",
        murmur="none"
    )
    result = validator.validate_cycle(normal_beat)
    print(f"  → Valid: {result.overall_valid}")
    print(f"  → Confidence: {result.overall_confidence:.2f}")
    print(f"  → Summary: {result.summary}")
    
    # Test 2: S1 too early (impossible)
    print("\n[TEST 2] S1 too early (10ms after R-peak)")
    early_s1 = create_test_beat(
        beat_index=2,
        r_peak=100,
        s1=110,  # Only 10ms after R-peak (should be 30-70ms)
        s2=390,
        hr=72,
        rhythm="normal",
        murmur="none"
    )
    result = validator.validate_cycle(early_s1)
    print(f"  → Valid: {result.overall_valid}")
    print(f"  → Confidence: {result.overall_confidence:.2f}")
    print(f"  → S1 Reason: {result.s1_result.reason}")
    
    # Test 3: Bradycardia with normal label (inconsistent)
    print("\n[TEST 3] Bradycardia labeled as 'normal'")
    brady_beat = create_test_beat(
        beat_index=3,
        r_peak=200,
        s1=250,
        s2=530,
        hr=45,  # Actually bradycardic
        rhythm="normal",  # But labeled normal (inconsistent!)
        murmur="none"
    )
    result = validator.validate_cycle(brady_beat)
    print(f"  → Valid: {result.overall_valid}")
    print(f"  → Rhythm Reason: {result.rhythm_result.reason}")
    
    # Test 4: Systolic murmur
    print("\n[TEST 4] Systolic murmur detected")
    murmur_beat = create_test_beat(
        beat_index=4,
        r_peak=300,
        s1=350,
        s2=650,
        hr=72,
        rhythm="normal",
        murmur="systolic"
    )
    result = validator.validate_cycle(murmur_beat)
    print(f"  → Valid: {result.overall_valid}")
    print(f"  → Murmur Reason: {result.murmur_result.reason}")
    
    # Test 5: AF (Atrial Fibrillation)
    print("\n[TEST 5] Atrial Fibrillation detected")
    af_beat = create_test_beat(
        beat_index=5,
        r_peak=400,
        s1=450,
        s2=730,
        hr=88,
        rhythm="afib",
        murmur="none"
    )
    result = validator.validate_cycle(af_beat)
    print(f"  → Valid: {result.overall_valid}")
    print(f"  → Rhythm Reason: {result.rhythm_result.reason}")
    if result.rhythm_result.warnings:
        print(f"  → Warnings: {result.rhythm_result.warnings}")
    
    # ===== UNITY RENDER COMMAND TEST =====
    print("\n" + "=" * 60)
    print("UNITY RENDER COMMAND GENERATION")
    print("=" * 60)
    
    print("\n[TEST] Generating Unity command for normal beat")
    normal_result = validator.validate_cycle(normal_beat)
    unity_cmd = validator.get_unity_command(normal_result)
    print(f"  → Render Confidence: {unity_cmd.render_confidence:.2f}")
    print(f"  → Show Rhythm: {unity_cmd.show_rhythm}")
    print(f"  → Show Murmur: {unity_cmd.show_murmur}")
    print(f"  → Highlight Valves: {unity_cmd.highlight_valves}")
    print(f"  → Play Haptic: {unity_cmd.play_haptic}")
    print(f"  → Caution Mode: {unity_cmd.caution_mode}")
    
    print("\n[TEST] Generating Unity command for invalid beat")
    invalid_result = validator.validate_cycle(early_s1)
    unity_cmd = validator.get_unity_command(invalid_result)
    print(f"  → Render Confidence: {unity_cmd.render_confidence:.2f}")
    print(f"  → Caution Mode: {unity_cmd.caution_mode}")
    print(f"  → Caution Reason: {unity_cmd.caution_reason}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
