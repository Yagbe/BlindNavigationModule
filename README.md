# Beacon — AI-Powered Mobility Assistant

Beacon is a mobile robot that **guides** a visually impaired user (not the other way around).  
It scouts the path, warns about obstacles, narrates the environment, and emits audio so the user can follow.

## Capabilities
- **Sees the environment** (YOLOv8 on dual camera support)
- **Understands speech** (VAD + faster-whisper ASR)
- **Speaks** (OpenAI TTS)
- **Plans** (GPT-4o mini returns `MOVE/STOP/SAY` commands)
- **Moves** (FS90R servos via **PCA9685** or **lgpio**)

## Hardware
- Raspberry Pi 5 **or** Jetson Orin Nano (set `BEACON_CAMERA` to `opencv` on Jetson)
- 1–2 wide-FOV cameras (Raspberry Pi Camera Modules supported)
- PCA9685 16ch PWM driver (recommended) or direct GPIO PWM
- 2× FS90R (continuous rotation) servos + wheels
- Battery + 5V servo power rail

## Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure
Set environment variables (or edit `config.py`). The most important:
```bash
export OPENAI_API_KEY="sk-..."
export BEACON_MOTOR="pca9685"      # or "lgpio" or "none"
export BEACON_CAMERA="picamera2"   # or "opencv"
# Calibrate your PCA9685 neutral ticks:
export BEACON_PCA_LEFT_NEUTRAL=372
export BEACON_PCA_RIGHT_NEUTRAL=140
```

## Run
```bash
python blind_guide_bot.py
# Preview: http://<bot-ip>:5000
```

## Notes
- **VAD framing fixed**: audio is resampled to **16 kHz** and chunked into exact **30 ms (480-sample)** frames before passing to WebRTC VAD (resolves “expected 512 got 480” issues).
- **Planner robust**: ignores malformed `MOVE(fwd, turn)` and keeps you safe.
- **Modular backends**: quickly switch motors/cameras via environment variables.
- **Safer defaults**: robot stops if no motion command in 8s.
