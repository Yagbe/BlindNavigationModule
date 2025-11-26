"""
High-level planner: formats inputs (vision + speech), calls GPT,
and dispatches MOVE / STOP / SAY intents.
"""
from __future__ import annotations
import json, time, threading
from typing import Deque
from collections import deque

from openai import OpenAI

import config
import motors
import vision
import speech

SYSTEM_PROMPT = (
    "You are Beacon-GPT. Respond with ONE line only:\n"
    "  MOVE(fwd,turn)    # floats -1..1\n"
    "  STOP()\n"
    "  SAY(\"text\")\n"
    "If you are unsure, use SAY with brief, helpful feedback."
)

def loop(stop_event: threading.Event) -> None:
    mem: Deque[str] = deque(maxlen=8)
    client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
    last_tts = ("", 0.0)

    while not stop_event.is_set():
        vis = vision.get_state()
        sp  = speech.get_state()

        # throttle if there is no fresh signal
        if time.time() - vis["time"] > 2 and time.time() - sp["time"] > 2:
            time.sleep(0.3); continue

        user = json.dumps({"vision": vis["objects"], "speech": sp["text"]})
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Memory: " + json.dumps(list(mem))},
            {"role": "user", "content": user}
        ]

        try:
            if not client:
                # Safe fallback: only narrate if OpenAI key missing
                text = "Guidance unavailable (missing OPENAI_API_KEY)."
                if time.time() - last_tts[1] > 5:
                    speech.say(text)
                    last_tts = (text, time.time())
                time.sleep(0.5)
                continue

            rsp = client.chat.completions.create(model=config.GPT_MODEL, messages=msgs,
                                                 temperature=0.2, max_tokens=40)
            cmd = rsp.choices[0].message.content.strip()
            mem.append(cmd)
            print(cmd)

            if cmd.startswith("MOVE(") and cmd.endswith(")"):
                try:
                    fwd_str, turn_str = cmd[5:-1].split(",")
                    fwd  = float(fwd_str)
                    turn = float(turn_str)
                except Exception:
                    # If model returned symbolic names, ignore
                    speech.say("I couldn't compute motion from that command.")
                    continue
                motors.send_twist(fwd, turn)

            elif cmd.startswith("STOP"):
                motors.stop()

            elif cmd.startswith("SAY(") and cmd.endswith(")"):
                # robustly strip surrounding quotes if present
                txt = cmd[4:-1].strip().strip('"').strip("'")
                if (txt != last_tts[0]) or (time.time() - last_tts[1] > 6):
                    speech.say(txt)
                    last_tts = (txt, time.time())

        except Exception as e:
            print("Planner-err:", e)

        time.sleep(0.5)
