#!/usr/bin/env python3
"""
Beacon - AI mobility assistant robot (clean, modular version)
- Vision: YOLOv8 + dual camera support
- Audio: VAD + faster-whisper ASR + OpenAI TTS
- Planner: GPT-4o mini
- Motors: PCA9685 or lgpio (select with BEACON_MOTOR)
- Web MJPEG preview at http://<ip>:5000
"""

from __future__ import annotations
import sys, time, threading
from flask import Flask, Response, stream_with_context

import config
import motors
import vision
import speech
import planner

app = Flask(__name__)

@app.route("/")
def index():
    return '<h2>Beacon camera</h2><img src="/stream.mjpg">'

@app.route("/stream.mjpg")
def stream():
    def gen():
        boundary = b"--frame\r\n"
        while True:
            jpg = vision.get_latest_jpg()
            if jpg:
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return Response(stream_with_context(gen()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():
    # init motors
    print(motors.init())

    # stop flag
    stop_event = threading.Event()

    # start threads
    threads = [
        threading.Thread(target=vision.loop,  args=(stop_event,), daemon=True),
        threading.Thread(target=speech.loop,  args=(stop_event,), daemon=True),
        threading.Thread(target=planner.loop, args=(stop_event,), daemon=True),
        threading.Thread(target=app.run, kwargs=dict(host=config.HTTP_HOST,
                                                     port=config.HTTP_PORT,
                                                     threaded=True),
                         daemon=True)
    ]
    for t in threads: t.start()

    print(f"Open  http://<Pi-or-Orin-IP>:{config.HTTP_PORT}  for live video")

    # watchdog: auto-stop wheels if no command for 8s
    try:
        while True:
            if time.time() - motors.last_move_ts() > 8.0:
                motors.stop()
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        motors.shutdown()
        print("-- Bye --")

if __name__ == "__main__":
    main()
