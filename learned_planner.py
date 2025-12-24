"""
Local on-device planner loop (experience-based).
Drop-in alternative to planner.py (GPT-based).

Enable it by setting:
  export BEACON_PLANNER=learned
or leaving OPENAI_API_KEY unset (see blind_guide_bot.py selection logic).

This planner:
- reads vision + speech states
- feeds them into LearningAI
- sends motor commands
- uses speech.say() for warnings/status
"""
from __future__ import annotations

import time, threading
from typing import Optional

import motors
import vision
import speech
import config

from learning_ai import LearningAI

def loop(stop_event: threading.Event) -> None:
    ai = LearningAI()
    last_tts = ("", 0.0)

    while not stop_event.is_set():
        vis = vision.get_state()
        sp  = speech.get_state()

        # if no fresh signals, keep loop light
        if time.time() - vis["time"] > 2 and time.time() - sp["time"] > 4:
            time.sleep(0.2)
            continue

        twist, say = ai.update(vis.get("objects", []), sp.get("text", ""))

        # safety: if we haven't issued motion recently, stop (handled also in main watchdog)
        motors.send_twist(twist.fwd, twist.turn)

        if say and ((say != last_tts[0]) or (time.time() - last_tts[1] > 5.5)):
            speech.say(say)
            last_tts = (say, time.time())

        time.sleep(0.15)
