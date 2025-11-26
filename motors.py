"""
Motor backends for Beacon.
- PCA9685 (preferred for FS90R continuous rotation servos)
- lgpio (direct GPIO servo pulses)
- none (simulation)
"""

from __future__ import annotations
from typing import Optional
import time

import config

# Exported API:
# - init()
# - send_twist(fwd, turn)  where both in [-1.0, 1.0]
# - stop()
# - shutdown()

_last_move_ts = 0.0

# -------------- Helper for clamping ----------------------------------
def _clamp(v: float, lo: float=-1.0, hi: float=1.0) -> float:
    return max(lo, min(hi, v))


# -------------- PCA9685 backend --------------------------------------
class _FS90R:
    """
    Continuous rotation servo wrapper for PCA9685 channel.
    'neutral' and 'span' MUST be calibrated per wheel.
    """
    def __init__(self, pca, channel: int, neutral: int, span: int=205, deadband: float=0.05):
        self._out = pca.channels[channel]
        self._neutral = int(neutral)
        self._span = int(span)
        self._deadband = float(deadband)

    def _ticks12(self, value: float) -> int:
        v = _clamp(value)
        if abs(v) < self._deadband:
            ticks = self._neutral
        else:
            ticks = self._neutral + int(v * self._span)
        return max(0, min(4095, ticks))

    def throttle(self, value: float) -> None:
        self._out.duty_cycle = self._ticks12(value) << 4

    def stop(self) -> None:
        self._out.duty_cycle = self._neutral << 4


class _PCA9685Backend:
    def __init__(self):
        import board, busio
        from adafruit_pca9685 import PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        self._pwm = PCA9685(i2c)
        self._pwm.frequency = 50  # 20 ms frame
        self.left  = _FS90R(self._pwm, config.PCA9685_LEFT_CH,
                            config.PCA9685_LEFT_NEUTRAL,  config.PCA9685_LEFT_SPAN)
        self.right = _FS90R(self._pwm, config.PCA9685_RIGHT_CH,
                            config.PCA9685_RIGHT_NEUTRAL, config.PCA9685_RIGHT_SPAN)

    def twist(self, fwd: float, turn: float):
        # Differential drive: mix forward & turn
        r = _clamp(+fwd - turn)
        l = _clamp(+fwd + turn)
        self.right.throttle(r)
        self.left.throttle(l)

    def stop(self):
        self.right.stop()
        self.left.stop()

    def shutdown(self):
        self.stop()


# -------------- lgpio backend (direct servo pulses) ------------------
class _LGPIOBackend:
    def __init__(self):
        import lgpio
        self._lg = lgpio
        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, config.LGPIO_LEFT_PIN, 0)
        lgpio.gpio_claim_output(self._h, config.LGPIO_RIGHT_PIN, 0)

    def _servo(self, pin: int, throttle: float):
        # Map [-1,1] to pulse width around 1500 us
        pulse = int(1500 + 400 * _clamp(throttle))
        self._lg.tx_servo(self._h, pin, pulse)

    def twist(self, fwd: float, turn: float):
        self._servo(config.LGPIO_RIGHT_PIN, +fwd - turn)
        self._servo(config.LGPIO_LEFT_PIN,  +fwd + turn)

    def stop(self):
        # Send neutral (1500us) to both
        self._lg.tx_servo(self._h, config.LGPIO_LEFT_PIN, 1500)
        self._lg.tx_servo(self._h, config.LGPIO_RIGHT_PIN, 1500)

    def shutdown(self):
        self.stop()


# -------------- Null backend -----------------------------------------
class _NullBackend:
    def twist(self, fwd: float, turn: float):
        pass
    def stop(self):
        pass
    def shutdown(self):
        pass


# ----------- Module-level facade -------------------------------------
_backend = None  # type: Optional[object]

def init() -> str:
    global _backend
    kind = config.MOTOR_BACKEND.lower()
    if kind == "pca9685":
        _backend = _PCA9685Backend()
        return "PCA9685 motor backend active"
    elif kind == "lgpio":
        _backend = _LGPIOBackend()
        return "lgpio motor backend active"
    else:
        _backend = _NullBackend()
        return "Motor backend disabled (simulation)"

def send_twist(fwd: float, turn: float) -> None:
    global _last_move_ts
    if _backend:
        _backend.twist(fwd, turn)
    _last_move_ts = time.time()

def stop() -> None:
    if _backend:
        _backend.stop()

def shutdown() -> None:
    if _backend:
        _backend.shutdown()

def last_move_ts() -> float:
    return _last_move_ts
