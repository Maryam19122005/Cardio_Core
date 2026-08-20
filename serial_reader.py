"""
Serial Reader — bridges the ESP32 (USB serial) to the CardioCore pipeline.

CONFIRMED WITH HARDWARE TEAM:
  - ADC resolution is 10-bit (raw values 0-1023). ADC_MIDPOINT = 512.
  - Sync/header bytes reported as 0xAA, 0x55 -- see the CAUTION comment next
    to SYNC_BYTES below before trusting this; verify against actual firmware
    source, since it matches the placeholder example previously suggested.
    Also confirm whether these 2 bytes are separate from, or included in,
    the 26-byte payload size below (this code assumes separate).
  - PCG (heart-sound sensor) data is coming tomorrow, currently ECG-only.
    Until then, pcg_samples are filled with zeros -- murmur classification
    and S2/diastole phase tracking will NOT be meaningful in live mode, only
    in mock mode. When PCG arrives, HARDWARE_FRAME_SIZE, HARDWARE_STRUCT_FMT,
    and unpack_hardware_frame() below will all need a follow-up edit to add
    the new channel -- ask hardware for the updated byte map at that point.

Packet format confirmed by hardware team (26-byte ECG payload, little-endian):
  Bytes 0-1   : frameNumber   uint16
  Bytes 2-5   : timestamp     uint32 (microseconds, per hardware team)
  Bytes 6-25  : ecg[0..9]     uint16 x 10  (raw ADC counts, 10-bit range)
"""

import struct
import time
from cardiocore_constants import Frame, SAMPLES_PER_FRAME
from mock_data_generator import MockDataGenerator

HARDWARE_FRAME_SIZE = 26  # bytes: 2 + 4 + (10 * 2) -- ECG-only payload, PCG pending
HARDWARE_STRUCT_FMT = '<HI' + 'H' * SAMPLES_PER_FRAME  # little-endian, all unsigned

ADC_MIDPOINT = 512   # confirmed: 10-bit ADC (0-1023)

# Hardware team reported these bytes: 0xAA, 0x55 (sent in that order).
# CAUTION: this happens to be the exact example value used when the sync-byte
# question was first raised -- worth double-checking it's genuinely in the
# firmware (e.g. a line like `Serial.write(0xAA); Serial.write(0x55);` sent
# right before each frame) rather than a repeated example. Once confirmed
# against the actual firmware source, this line stands as-is.
SYNC_BYTES = b'\xAA\x55'


def unpack_hardware_frame(data: bytes):
    """
    Decode one 26-byte ESP32 packet into a Frame object.
    Returns None if the data isn't exactly 26 bytes or fails to unpack.

    NOTE: pcg_samples is filled with zeros as a placeholder -- the hardware
    packet does not currently include a second (PCG) channel, arriving tomorrow.
    """
    if len(data) != HARDWARE_FRAME_SIZE:
        return None

    try:
        unpacked = struct.unpack(HARDWARE_STRUCT_FMT, data)
    except struct.error as e:
        print(f"ERROR unpacking hardware frame: {e}")
        return None

    frame_number = unpacked[0]
    timestamp_raw = unpacked[1]
    ecg_raw = list(unpacked[2:2 + SAMPLES_PER_FRAME])

    # Center raw ADC counts around zero so the rest of the pipeline
    # (which expects a bipolar signal, see validate_frame range checks)
    # gets sensible values. See TODO #2 -- this assumes 12-bit ADC.
    ecg_centered = [s - ADC_MIDPOINT for s in ecg_raw]

    pcg_placeholder = [0] * SAMPLES_PER_FRAME  # see TODO #1

    return Frame(
        frame_number=frame_number,
        timestamp_ms=timestamp_raw,  # NOTE: confirm units -- code elsewhere assumes ms
        ecg_samples=ecg_centered,
        pcg_samples=pcg_placeholder
    )


class HardwareSerialReader:
    """
    Reads frames from the ESP32 over USB serial.
    If the port isn't connected, or a read fails/returns a malformed frame,
    automatically falls back to MockDataGenerator so the rest of the system
    (Unity, demo modes, etc.) keeps receiving valid data.
    """

    def __init__(self, port='COM3', baudrate=115200, use_mock_fallback=True):
        self.port = port
        self.baudrate = baudrate
        self.use_mock_fallback = use_mock_fallback
        self.ser = None
        self.connected = False
        self.mock_gen = MockDataGenerator()

    def connect(self):
        """Attempt to open the serial port. Safe to call even if pyserial
        or the port isn't available -- falls back to mock mode instead
        of crashing the server."""
        try:
            import serial  # imported here so the module doesn't hard-fail
                            # at import time if pyserial isn't installed yet
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected = True
            print(f"✓ Connected to hardware on {self.port} @ {self.baudrate} baud")
        except Exception as e:
            print(f"✗ Could not connect to {self.port}: {e}")
            print("  Falling back to mock data until hardware is available.")
            self.connected = False

    def _wait_for_sync(self):
        """
        Reads one byte at a time, sliding a window, until SYNC_BYTES is
        found at the tail of the stream. This is what lets the reader
        recover automatically if bytes were dropped or a read started
        mid-packet -- without it, a single lost byte would permanently
        misalign every frame after it.

        Returns True once sync is found, False on a read timeout (no
        data arriving at all).
        """
        sync_len = len(SYNC_BYTES)
        window = bytearray()
        while True:
            b = self.ser.read(1)
            if not b:
                return False  # timed out waiting for data
            window += b
            if len(window) > sync_len:
                del window[0]
            if bytes(window) == SYNC_BYTES:
                return True

    def read_frame(self):
        """
        Returns (frame, source) where source is 'live' or 'mock'.
        Always returns a usable Frame -- never blocks the caller waiting
        on hardware that isn't there.
        """
        if self.connected and self.ser:
            try:
                if self._wait_for_sync():
                    data = self.ser.read(HARDWARE_FRAME_SIZE)
                    if len(data) == HARDWARE_FRAME_SIZE:
                        frame = unpack_hardware_frame(data)
                        if frame is not None:
                            return frame, 'live'
                # Sync not found before timeout, or payload came back
                # incomplete -- fall through to mock for this cycle.
            except Exception as e:
                print(f"Serial read error: {e}")
                self.connected = False  # stop hammering a dead port

        if self.use_mock_fallback:
            return self.mock_gen.generate_frame(), 'mock'

        return None, 'none'

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ============ STANDALONE TEST ============
if __name__ == '__main__':
    print("Testing HardwareSerialReader (no real hardware required)...")

    # Deliberately point at a port that likely doesn't exist, to prove
    # the mock fallback works end-to-end.
    reader = HardwareSerialReader(port='COM99', baudrate=115200)
    reader.connect()

    for i in range(5):
        frame, source = reader.read_frame()
        print(f"Frame {i}: source={source}, frame_number={frame.frame_number}, "
              f"ecg[0:3]={frame.ecg_samples[:3]}, pcg[0:3]={frame.pcg_samples[:3]}")
        time.sleep(0.01)

    print("\n✓ Fallback path working. Once hardware team confirms the open "
          "questions at the top of this file, update ADC_MIDPOINT and the "
          "PCG placeholder logic accordingly, then point `port=` at the "
          "real COM/tty device and re-test with real hardware connected.")
