"""
twin_contract.py
Adapter: maps CardioFusionEngine's internal TwinState to the FROZEN /data contract
Unity reads. Ramish owns /data; this is the single place server output is shaped to it.
Nothing upstream (fusion / AI / buffers) changes.

FROZEN /data (never rename):
    bpm, phase, r_peak_t, r_s1_ms, s1_s2_ms,
    rhythm_label, rhythm_conf, murmur_label, murmur_conf,
    lead_off, ecg[], pcg_env[]
"""

from collections import deque
from typing import Any, Dict, List, Optional

FALLBACK_R_S1_MS = 50.0
FALLBACK_S1_S2_MS = 300.0
R_S1_GATE = (5.0, 250.0)
S1_S2_GATE = (100.0, 600.0)

RIBBON_SECONDS = 1.5
RIBBON_POINTS = 180
ENVELOPE_SMOOTH = 20


class TwinContractAdapter:
    def __init__(self, sample_rate: int = 1000):
        self.sample_rate = sample_rate
        self.session_start_ms: Optional[float] = None
        self._last_r_s1 = FALLBACK_R_S1_MS
        self._last_s1_s2 = FALLBACK_S1_S2_MS

    def build(self, state, buffers) -> Dict[str, Any]:
        ecg_win, pcg_win = buffers.get_last_n_samples(int(RIBBON_SECONDS * self.sample_rate))
        conf = float(getattr(state, "overall_confidence", 1.0))
        return {
            "bpm": float(state.bpm),
            "phase": self._wrap01(state.cardiac_phase),
            "r_peak_t": self._session_rel(state.last_r_peak_ms),
            "r_s1_ms": self._r_s1(state),
            "s1_s2_ms": self._s1_s2(state),
            "rhythm_label": state.rhythm.value,
            "rhythm_conf": round(conf, 3),
            "murmur_label": state.murmur.value,
            "murmur_conf": round(conf, 3),
            "lead_off": bool(state.lead_off),
            "ecg": self._decimate(ecg_win, RIBBON_POINTS),
            "pcg_env": self._decimate(self._envelope(pcg_win, ENVELOPE_SMOOTH), RIBBON_POINTS),
        }

    def _session_rel(self, abs_ms: float) -> float:
        if abs_ms is None or abs_ms < 0:
            return 0.0
        if self.session_start_ms is None:
            self.session_start_ms = abs_ms
        return round(max(0.0, abs_ms - self.session_start_ms), 1)

    def _r_s1(self, state) -> float:
        d = state.last_s1_ms - state.last_r_peak_ms
        if R_S1_GATE[0] <= d <= R_S1_GATE[1]:
            self._last_r_s1 = d
        return round(self._last_r_s1, 1)

    def _s1_s2(self, state) -> float:
        d = state.last_s2_ms - state.last_s1_ms
        if S1_S2_GATE[0] <= d <= S1_S2_GATE[1]:
            self._last_s1_s2 = d
        return round(self._last_s1_s2, 1)

    @staticmethod
    def _wrap01(x: float) -> float:
        x = float(x) % 1.0
        return x + 1.0 if x < 0 else x

    @staticmethod
    def _decimate(samples: List[float], n_out: int) -> List[float]:
        if not samples:
            return []
        if len(samples) <= n_out:
            return [round(float(s), 2) for s in samples]
        step = len(samples) / n_out
        return [round(float(samples[int(i * step)]), 2) for i in range(n_out)]

    @staticmethod
    def _envelope(samples: List[float], smooth: int) -> List[float]:
        if not samples:
            return []
        win: deque = deque(maxlen=max(1, smooth))
        out: List[float] = []
        for s in samples:
            win.append(abs(float(s)))
            out.append(sum(win) / len(win))
        return out
