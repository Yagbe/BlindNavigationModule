"""
Speech: VAD-framed microphone capture + faster-whisper ASR + OpenAI TTS.
We keep exact 30 ms frames at 16 kHz to satisfy WebRTC VAD.
"""
from __future__ import annotations
import io, time, math, threading, queue
from typing import Dict, Any

import numpy as np
import sounddevice as sd, soundfile as sf
import webrtcvad
import soxr

from faster_whisper import WhisperModel
from openai import OpenAI

import config

# ------------------ Shared state -------------------------------------
_state = {"time": 0.0, "text": ""}
_lock = threading.Lock()

def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)

def _put_text(txt: str):
    with _lock:
        _state["time"] = time.time()
        _state["text"] = txt

# ------------------ Audio devices ------------------------------------
def _select_mic():
    # choose first device with input channels > 0
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            return i, int(dev["default_samplerate"])
    raise RuntimeError("No input audio devices found")

MIC_DEV, MIC_SR = _select_mic()
FRAME_SAMPLES_16K = int(config.TARGET_SR * (config.BLOCK_MS / 1000.0))  # 480 for 30ms

# ------------------ VAD ----------------------------------------------
_vad = webrtcvad.Vad(config.VAD_MODE)

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x*x) + 1e-12))

def _resample_to_16k(x: np.ndarray) -> np.ndarray:
    if MIC_SR == config.TARGET_SR:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x.astype(np.float32, copy=False), MIC_SR, config.TARGET_SR)

def _frame_to_pcm16(frame_f32_16k: np.ndarray) -> bytes:
    # WebRTC expects bytes PCM16 with exact length (10/20/30 ms)
    f = np.clip(frame_f32_16k, -1.0, 1.0)
    return (f * 32767.0).astype(np.int16, copy=False).tobytes()

def _vad_accepts(frame_f32_16k: np.ndarray) -> bool:
    if _rms(frame_f32_16k) < config.NOISE_FLOOR:
        return False
    try:
        return _vad.is_speech(_frame_to_pcm16(frame_f32_16k), config.TARGET_SR)
    except webrtcvad.Error:
        return False

# ------------------ ASR model ----------------------------------------
_asr = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")  # CPU-friendly

# ------------------ TTS (OpenAI) -------------------------------------
def say(text: str):
    try:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        audio = client.audio.speech.create(model="tts-1", voice="alloy", input=text).content
        with sf.SoundFile(io.BytesIO(audio)) as f:
            sd.play(f.read(dtype="float32"), f.samplerate, blocking=True)
    except Exception as e:
        print("TTS-err:", e)
    print(text)

# ------------------ Capture thread -----------------------------------
def loop(stop_event: threading.Event) -> None:
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)

    def _mic_cb(indata, frames, t, status):
        if not audio_q.full():
            audio_q.put(indata.copy())

    blocksz = int(MIC_SR * (config.BLOCK_MS / 1000.0))
    print(f"Mic@{MIC_SR}Hz ({blocksz}frames)  device#{MIC_DEV}")

    # rolling buffer to build exact 30ms frames at 16k
    hold = np.empty((0,), dtype=np.float32)

    with sd.InputStream(device=MIC_DEV, samplerate=MIC_SR, channels=1,
                        dtype="float32", blocksize=blocksz, callback=_mic_cb):
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.5).flatten().astype(np.float32, copy=False)
            except queue.Empty:
                continue

            # resample immediately to 16k
            res = _resample_to_16k(chunk)
            hold = np.concatenate([hold, res])

            # while we have at least one full VAD frame, process
            while len(hold) >= FRAME_SAMPLES_16K:
                frame = hold[:FRAME_SAMPLES_16K]
                hold  = hold[FRAME_SAMPLES_16K:]

                if not _vad_accepts(frame):
                    continue

                # voice detected -> collect ~1.0s more at 16k
                buf = [frame]
                start = time.time()
                while time.time() - start < 1.0 and not stop_event.is_set():
                    # make sure we have enough; if not, break and wait next loop
                    if len(hold) < FRAME_SAMPLES_16K:
                        break
                    buf.append(hold[:FRAME_SAMPLES_16K])
                    hold = hold[FRAME_SAMPLES_16K:]

                mono = np.concatenate(buf, axis=0)

                # Transcribe (single chunk)
                try:
                    segs, _ = _asr.transcribe(mono, language="en")
                    text = "".join(s.text for s in segs).strip()
                except Exception as e:
                    print("Whisper-err:", e)
                    text = ""

                if text:
                    _put_text(text)
                    print("[You]", text)
