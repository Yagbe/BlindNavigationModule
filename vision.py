"""
Vision thread: grabs frames, runs YOLO, publishes top-K objects.
Also maintains an MJPEG buffer for live web preview.
"""
from __future__ import annotations
from typing import Dict, List, Any
import time
import threading

import cv2
from ultralytics import YOLO

import config

_latest_jpg = b""
_state = {"time": 0.0, "objects": []}
_lock = threading.Lock()

def get_latest_jpg() -> bytes:
    return _latest_jpg

def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)

def _put_state(objects: List[Dict[str, Any]]):
    with _lock:
        _state["time"] = time.time()
        _state["objects"] = objects[:config.TOP_K_OBJECTS]

def _encode_jpeg(bgr) -> None:
    global _latest_jpg
    ok, jpg = cv2.imencode(".jpg", bgr)
    if ok:
        _latest_jpg = jpg.tobytes()

def _open_camera():
    if config.CAMERA_BACKEND == "picamera2":
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"size": (config.CAM_RES_W, config.CAM_RES_H)}
        ))
        cam.start()
        return ("picamera2", cam)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAM_RES_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_RES_H)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera with OpenCV")
        return ("opencv", cap)

def loop(stop_event: threading.Event) -> None:
    backend, cam = _open_camera()
    model = YOLO(config.YOLO_MODEL)

    while not stop_event.is_set():
        if backend == "picamera2":
            bgr = cam.capture_array()
        else:
            ok, bgr = cam.read()
            if not ok:
                time.sleep(0.01); continue

        rgb = bgr[:, :, ::-1]
        results = model(rgb, verbose=False)[0]

        objects = []
        for box in results.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            x_center_px = float(box.xywh[0][0])
            cx = round(x_center_px / float(rgb.shape[1]), 2)
            objects.append({"label": label,
                            "conf": round(conf, 2),
                            "cx": cx})
        objects.sort(key=lambda o: o["conf"], reverse=True)

        # preview overlay (optional)
        if config.SHOW_PREVIEW:
            for i, o in enumerate(objects[:config.TOP_K_OBJECTS]):
                cv2.putText(bgr, f'{o["label"]}:{o["conf"]:.2f}',
                            (10, 30 + 25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Beacon-cam", bgr); cv2.waitKey(1)

        _put_state(objects)
        _encode_jpeg(bgr)

    # cleanup
    if backend == "picamera2":
        cam.close()
    else:
        cam.release()
    if config.SHOW_PREVIEW:
        cv2.destroyAllWindows()
