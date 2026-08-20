"""
serial_reader.py
Reads real ESP32 frames from USB serial and feeds them into CardioFusionEngine.
Replaces the mock in the server's 'live' mode. Run on the laptop the Quest talks to.

FRAME FORMAT NOTE (the ESP32<->Python decision Umer & Kashif must lock):
This reader expects the 46-byte Frame that FrameParser already parses and the
engine already consumes (tested):
    [frame_num:uint16][timestamp_ms:uint32][ECG:10 x int16][PCG:10 x int16]  little-endian
If the firmware instead emits the AA55 single-sample contract packet
(AA55|type|t_us|sample|flags|crc8), either (a) have the firmware batch 10+10
samples into this 46-byte frame before sending, or (b) write a stream assembler
here that syncs on AA55, checks crc8, demuxes E/P, and builds Frames. Path (a) is
faster for the demo because the entire Python side is already tested against it.
"""

import threading
import serial  # pip install pyserial
from signal_buffers import FrameParser

FRAME_BYTES = 46


class SerialFrameReader:
    def __init__(self, engine, server_state, data_lock, port, baud=921600):
        self.engine = engine
        self.server_state = server_state
        self.data_lock = data_lock
        self.port = port
        self.baud = baud
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"✓ Serial reader started on {self.port} @ {self.baud}")

    def stop(self):
        self._running = False

    def _loop(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            print(f"SERIAL OPEN FAILED ({self.port}): {e} -- staying on mock")
            return

        buf = bytearray()
        while self._running and self.server_state['running']:
            try:
                chunk = ser.read(FRAME_BYTES)
                if not chunk:
                    continue
                buf.extend(chunk)
                while len(buf) >= FRAME_BYTES:
                    raw = bytes(buf[:FRAME_BYTES])
                    del buf[:FRAME_BYTES]
                    frame = FrameParser.unpack_frame(raw)
                    if frame is None:
                        buf.clear()  # desynced; drop and resync on next reads
                        continue
                    state = self.engine.process_frame(frame)
                    if state:
                        with self.data_lock:
                            self.server_state['last_state'] = state
                            self.server_state['frame_count'] += 1
            except Exception as e:
                with self.data_lock:
                    self.server_state['error_count'] += 1
                print(f"serial read error: {e}")
        try:
            ser.close()
        except Exception:
            pass
