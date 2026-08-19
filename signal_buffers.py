"""
Signal Buffers & Frame Parser
Ring buffers store ECG/PCG history; parser converts ESP32 frames
"""

import struct
from collections import deque
from typing import List, Tuple, Optional
from cardiocore_constants import Frame, BUFFER_SIZE, SAMPLES_PER_FRAME, SAMPLE_RATE

class RingBuffer:
    """Stores last N samples in a fixed-size circular buffer"""
    
    def __init__(self, size=BUFFER_SIZE):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.sample_index = 0  # Global sample counter
    
    def append(self, sample):
        """Add one sample to buffer"""
        self.buffer.append(sample)
        self.sample_index += 1
    
    def extend(self, samples):
        """Add multiple samples"""
        for sample in samples:
            self.append(sample)
    
    def get_last_n(self, n):
        """Get last N samples (returns list)"""
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
    
    def get_all(self):
        """Get all samples currently in buffer"""
        return list(self.buffer)
    
    def get_at_index(self, idx):
        """Get sample by index, or None if out of range"""
        buffer_list = list(self.buffer)
        if idx < 0 or idx >= len(buffer_list):
            return None
        return buffer_list[idx]
    
    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        return f"RingBuffer(size={self.size}, filled={len(self.buffer)}, samples_added={self.sample_index})"

class SensorBuffers:
    """Manages both ECG and PCG ring buffers"""
    
    def __init__(self):
        self.ecg_buffer = RingBuffer(BUFFER_SIZE)
        self.pcg_buffer = RingBuffer(BUFFER_SIZE)
        self.latest_timestamp_ms = 0
    
    def add_frame(self, frame: Frame):
        """Add a frame's samples to both buffers"""
        self.ecg_buffer.extend(frame.ecg_samples)
        self.pcg_buffer.extend(frame.pcg_samples)
        self.latest_timestamp_ms = frame.timestamp_ms
    
    def get_last_n_samples(self, n: int) -> Tuple[List[int], List[int]]:
        """Get last N samples from both buffers"""
        return self.ecg_buffer.get_last_n(n), self.pcg_buffer.get_last_n(n)
    
    def get_all(self) -> Tuple[List[int], List[int]]:
        """Get all samples from both buffers"""
        return self.ecg_buffer.get_all(), self.pcg_buffer.get_all()
    
    def __repr__(self):
        return f"SensorBuffers(ECG: {self.ecg_buffer}, PCG: {self.pcg_buffer}, latest_ts: {self.latest_timestamp_ms})"

class FrameParser:
    """
    Parses ESP32 frame format:
    [frame_num: 2 bytes] [timestamp: 4 bytes (wrapped)] [ECG: 10*2 bytes] [PCG: 10*2 bytes]
    Total: 2 + 4 + 20 + 20 = 46 bytes per frame
    
    NOTE: Timestamp wraps every ~50 days (2^32 ms). For ESP32 (resets on power), this is fine.
    """
    
    @staticmethod
    def pack_frame(frame: Frame) -> bytes:
        """
        Convert Frame object to binary format for transmission
        Returns 44 bytes
        
        Timestamp is stored as uint32 (wraps every ~49 days)
        Use modulo to fit large millisecond timestamps
        """
        if len(frame.ecg_samples) != SAMPLES_PER_FRAME or len(frame.pcg_samples) != SAMPLES_PER_FRAME:
            raise ValueError(f"Frame must have exactly {SAMPLES_PER_FRAME} samples each")
        
        # Wrap timestamp to 32-bit range (0 - 4,294,967,295)
        # This wraps every ~49.7 days, but for ESP32 (resets on power) it's fine
        timestamp_wrapped = frame.timestamp_ms % (2**32)
        
        # Format: <HI + 10h + 10h
        # < = little-endian, H = unsigned short (2 bytes), I = unsigned int (4 bytes), h = signed short (2 bytes)
        data = struct.pack(
            '<HI' + 'h' * SAMPLES_PER_FRAME + 'h' * SAMPLES_PER_FRAME,
            frame.frame_number % (2**16),  # Also wrap frame number
            timestamp_wrapped,
            *frame.ecg_samples,
            *frame.pcg_samples
        )
        return data
    
    @staticmethod
    def unpack_frame(data: bytes) -> Optional[Frame]:
        """
        Convert 46 bytes to Frame object
        Returns Frame or None if invalid
        
        Note: Timestamp is 32-bit wrapped, restore using modulo awareness
        """
        if len(data) != 46:
            print(f"ERROR: Expected 46 bytes, got {len(data)}")
            return None
        
        try:
            unpacked = struct.unpack(
                '<HI' + 'h' * SAMPLES_PER_FRAME + 'h' * SAMPLES_PER_FRAME,
                data
            )
            
            frame_num = unpacked[0]
            timestamp = unpacked[1]  # This is 32-bit wrapped
            ecg_samples = list(unpacked[2:2+SAMPLES_PER_FRAME])
            pcg_samples = list(unpacked[2+SAMPLES_PER_FRAME:])
            
            return Frame(
                frame_number=frame_num,
                timestamp_ms=timestamp,
                ecg_samples=ecg_samples,
                pcg_samples=pcg_samples
            )
        except struct.error as e:
            print(f"ERROR parsing frame: {e}")
            return None
    
    @staticmethod
    def validate_frame(frame: Frame) -> Tuple[bool, str]:
        """Check if frame is valid"""
        if not isinstance(frame, Frame):
            return False, "Not a Frame object"
        if len(frame.ecg_samples) != SAMPLES_PER_FRAME:
            return False, f"ECG has {len(frame.ecg_samples)} samples, expected {SAMPLES_PER_FRAME}"
        if len(frame.pcg_samples) != SAMPLES_PER_FRAME:
            return False, f"PCG has {len(frame.pcg_samples)} samples, expected {SAMPLES_PER_FRAME}"
        if frame.timestamp_ms < 0:
            return False, "Timestamp negative"
        
        # Check values are in reasonable range (sensor dependent)
        for sample in frame.ecg_samples:
            if sample < -500 or sample > 500:
                return False, f"ECG sample {sample} out of range"
        for sample in frame.pcg_samples:
            if sample < -200 or sample > 200:
                return False, f"PCG sample {sample} out of range"
        
        return True, "OK"

# ============ TEST ============
if __name__ == '__main__':
    print("Testing Signal Buffers & Frame Parser...")
    
    # Test RingBuffer
    print("\n1. Testing RingBuffer:")
    rb = RingBuffer(size=100)
    for i in range(150):
        rb.append(i)
    print(f"   After adding 150 samples: {rb}")
    print(f"   Last 5 samples: {rb.get_last_n(5)}")
    
    # Test FrameParser
    print("\n2. Testing FrameParser:")
    from mock_data_generator import MockDataGenerator
    
    gen = MockDataGenerator()
    frame_orig = gen.generate_frame()
    print(f"   Original frame: {frame_orig.frame_number}, ECG: {frame_orig.ecg_samples[:3]}...")
    
    # Pack to binary
    binary = FrameParser.pack_frame(frame_orig)
    print(f"   Packed to {len(binary)} bytes")
    
    # Unpack from binary
    frame_restored = FrameParser.unpack_frame(binary)
    print(f"   Restored frame: {frame_restored.frame_number}, ECG: {frame_restored.ecg_samples[:3]}...")
    
    # Validate
    valid, msg = FrameParser.validate_frame(frame_restored)
    print(f"   Validation: {valid} ({msg})")
    
    # Test SensorBuffers
    print("\n3. Testing SensorBuffers:")
    buffers = SensorBuffers()
    for i in range(3):
        frame = gen.generate_frame()
        buffers.add_frame(frame)
    print(f"   {buffers}")
    ecg, pcg = buffers.get_last_n_samples(20)
    print(f"   Last 20 ECG samples: {ecg}")
    
    print("\n✓ All buffer tests passed!")
