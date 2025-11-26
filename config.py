"""
Central configuration for Beacon robot.
Edit these values or override via environment variables.
"""
import os

# --- Vision / Cameras ---
CAM_RES_W = int(os.getenv("BEACON_CAM_W", 640))
CAM_RES_H = int(os.getenv("BEACON_CAM_H", 480))
TOP_K_OBJECTS = int(os.getenv("BEACON_TOPK", 6))

# Choose camera stack: "picamera2" (Raspberry Pi) or "opencv" (generic/Jetson)
CAMERA_BACKEND = os.getenv("BEACON_CAMERA", "picamera2")

# YOLO model (Ultralytics). "yolov8n.pt" is fast and light.
YOLO_MODEL = os.getenv("BEACON_YOLO", "yolov8n.pt")

# --- Audio ---
TARGET_SR = 16000                 # WebRTC VAD requires 8/16/32 kHz; we use 16 kHz
BLOCK_MS  = 30                    # frame size for VAD (10/20/30 ms valid); we use 30 ms
VAD_MODE  = int(os.getenv("BEACON_VAD_MODE", 2))   # 0=aggressive,3=sensitive
NOISE_FLOOR = float(os.getenv("BEACON_NOISE_FLOOR", "0.015"))  # RMS gate

# --- Speech backends ---
# Whisper: we use faster-whisper (no PyTorch required)
WHISPER_MODEL = os.getenv("BEACON_WHISPER", "small")  # "tiny"/"base"/"small" on CPU; "small" is a good balance

# Optional Porcupine wake word (set access key to enable)
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "")  # leave empty to disable
PORCUPINE_KEYWORDS   = os.getenv("PORCUPINE_KEYWORDS", "picovoice").split(",")

# --- Planner (OpenAI) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("BEACON_GPT", "gpt-4o-mini")

# --- Motors ---
# Backend: "pca9685", "lgpio" (PWM servo via GPIO), or "none"
MOTOR_BACKEND = os.getenv("BEACON_MOTOR", "pca9685")
# PCA9685 channel/neutral/span per wheel (continuous rotation FS90R)
# You MUST calibrate neutrals for your wheels.
PCA9685_LEFT_CH  = int(os.getenv("BEACON_PCA_LEFT_CH", 2))
PCA9685_RIGHT_CH = int(os.getenv("BEACON_PCA_RIGHT_CH", 1))
PCA9685_LEFT_NEUTRAL  = int(os.getenv("BEACON_PCA_LEFT_NEUTRAL", 372))
PCA9685_RIGHT_NEUTRAL = int(os.getenv("BEACON_PCA_RIGHT_NEUTRAL", 140))
PCA9685_LEFT_SPAN  = int(os.getenv("BEACON_PCA_LEFT_SPAN", 205))
PCA9685_RIGHT_SPAN = int(os.getenv("BEACON_PCA_RIGHT_SPAN", 120))

# lgpio pins (if you wire servos directly)
LGPIO_LEFT_PIN  = int(os.getenv("BEACON_LGPIO_LEFT", "12"))
LGPIO_RIGHT_PIN = int(os.getenv("BEACON_LGPIO_RIGHT", "13"))

# --- Web preview ---
HTTP_HOST = os.getenv("BEACON_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("BEACON_HTTP_PORT", "5000"))

# Show a local OpenCV window if DISPLAY is available and PREVIEW=1
SHOW_PREVIEW = bool(int(os.getenv("PREVIEW", "0")))
