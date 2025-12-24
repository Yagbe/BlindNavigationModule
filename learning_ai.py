"""
On-device learning navigation AI for Beacon.

Inputs:
- vision objects from vision.get_state()["objects"]
- speech transcripts from speech.get_state()["text"]
- optional pose estimates (Pose2D). If you don't have odometry yet, pass None:
  the planner will still do reactive obstacle avoidance.

Outputs:
- Twist2D (fwd, turn) in normalized [-1,1]
- optional speech string to announce (warnings / confirmations)

This module is designed to be swapped in for planner.py when you want
a fully local AI (no OpenAI API required).
"""
from __future__ import annotations

import json, math, os, time, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

import config
from nav_types import Pose2D, Twist2D, Obstacle
from experience import ExperienceMap

# -------------- utilities -----------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _wrap_pi(a: float) -> float:
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _softsign(x: float) -> float:
    return x / (1.0 + abs(x))

def _focal_px_from_fov(width_px: int, hfov_deg: float) -> float:
    # f = w/(2*tan(hfov/2))
    return float(width_px) / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))

# -------------- waypoint storage -----------------
def load_waypoints(path: str) -> Dict[str, Tuple[float, float]]:
    try:
        data = json.loads(open(path, "r", encoding="utf-8").read())
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in data.items():
        try:
            out[str(k).lower()] = (float(v["x"]), float(v["y"]))
        except Exception:
            continue
    return out

# -------------- perception from YOLO boxes -----------------
# Typical COCO class names we treat as obstacles.
OBSTACLE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "chair", "bench", "couch", "potted plant", "dog", "cat",
    "stop sign", "parking meter", "fire hydrant",
    # custom labels you might add with a fine-tuned YOLO model:
    "pothole", "hole", "crack", "curb", "stairs", "step", "cone", "bollard"
}

# nominal physical heights (meters) for rough monocular distance estimates
NOMINAL_HEIGHT_M = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 3.00,
    "dog": 0.55,
    "cat": 0.30,
    "chair": 0.90,
    "bench": 0.90,
    "couch": 1.00,
}

def yolo_objects_to_obstacles(objs: List[Dict[str, Any]],
                             img_w: int = None,
                             img_h: int = None) -> List[Obstacle]:
    """
    Convert YOLO object dicts from vision.py into obstacles in robot frame.

    vision.py currently publishes each object as:
      {"name": "...", "conf": 0.92, "xyxy": [x1,y1,x2,y2]}
    """
    img_w = int(img_w if img_w else config.CAM_RES_W)
    img_h = int(img_h if img_h else config.CAM_RES_H)

    f_px = _focal_px_from_fov(img_w, config.CAM_HFOV_DEG)

    obstacles: List[Obstacle] = []
    for o in objs or []:
        try:
            name = str(o.get("name", "unknown"))
            conf = float(o.get("conf", 0.0))
            if name not in OBSTACLE_CLASSES:
                continue
            x1, y1, x2, y2 = [float(v) for v in o.get("xyxy", [0,0,0,0])]
            cx = (x1 + x2) / 2.0
            h = max(1.0, (y2 - y1))
            # bearing from image center
            x_norm = (cx - img_w / 2.0)
            bearing = math.atan2(x_norm, f_px)  # rad

            # distance heuristic
            H = NOMINAL_HEIGHT_M.get(name, 1.0)
            # Use vertical geometry if VFOV set; otherwise rely on hfov focal.
            # This is crude but workable for obstacle avoidance.
            dist = (H * f_px) / h
            dist = _clamp(dist, 0.2, 12.0)
            obstacles.append(Obstacle(bearing=bearing, distance=dist, kind=name, confidence=conf, severity=1.0))
        except Exception:
            continue
    return obstacles

def detect_crowd(obstacles: List[Obstacle]) -> Tuple[bool, float]:
    """
    Returns (is_crowd, crowd_dist_m) based on persons ahead.
    """
    people = [o for o in obstacles if o.kind == "person"]
    if len(people) < config.CROWD_COUNT:
        return (False, 999.0)
    # focus on forward cone +/- 35 deg
    fwd_people = [p for p in people if abs(p.bearing) < math.radians(35)]
    if len(fwd_people) < config.CROWD_COUNT:
        return (False, 999.0)
    dist = float(np.median([p.distance for p in fwd_people]))
    return (dist < config.CROWD_DIST_M, dist)

# -------------- local planner -----------------
@dataclass
class Goal:
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    active: bool = False

class LearningAI:
    """
    Main orchestrator: parses speech -> destination, updates experience map,
    and outputs motion commands.
    """
    def __init__(self):
        self.exp = ExperienceMap()
        self.exp.load(config.EXPERIENCE_DB_PATH)

        self.waypoints = load_waypoints(config.WAYPOINTS_PATH)
        self.goal = Goal(active=False)

        self._last_spoken: Tuple[str, float] = ("", 0.0)
        self._last_text: str = ""
        self._last_save_t = time.time()
        self._pose = Pose2D()
        self._has_pose = False

        # simple "progress" memory: last chosen side around a crowd
        self._prefer_left = True

    # ---- public API ----
    def set_pose(self, pose: Pose2D) -> None:
        self._pose = pose
        self._has_pose = True

    def update(self, vision_objects: List[Dict[str, Any]], speech_text: str,
               pose: Optional[Pose2D] = None) -> Tuple[Twist2D, Optional[str]]:
        """
        One control step.
        """
        if pose is not None:
            self.set_pose(pose)

        say = self._handle_speech(speech_text or "")

        obstacles = yolo_objects_to_obstacles(vision_objects or [])
        self._learn_from_observations(obstacles)

        twist, warn = self._compute_action(obstacles)

        # traverse update (learning)
        if self._has_pose and abs(twist.fwd) > 0.05:
            self.exp.mark_traversed(self._pose, success=True)

        # persist occasionally
        if time.time() - self._last_save_t > 8.0:
            self.exp.save(config.EXPERIENCE_DB_PATH)
            self._last_save_t = time.time()

        # prefer warning speech over other speech
        if warn:
            say = warn
        return twist, say

    def debug_snapshot(self) -> Dict[str, Any]:
        return {
            "goal": {"active": self.goal.active, "name": self.goal.name, "x": self.goal.x, "y": self.goal.y},
            "experience": self.exp.snapshot(),
        }

    # ---- speech parsing ----
    def _handle_speech(self, txt: str) -> Optional[str]:
        txt = txt.strip()
        if not txt or txt == self._last_text:
            return None
        self._last_text = txt
        t = txt.lower()

        if any(w in t for w in ["stop", "halt", "pause"]):
            self.goal.active = False
            return "Stopping."

        # "take me to kitchen" / "go to front door"
        m = re.search(r"(take me to|go to|navigate to|destination)\s+(.+)$", t)
        if m:
            place = m.group(2).strip().lower()
            # remove filler words
            place = re.sub(r"\b(the|a|an|please)\b", "", place).strip()
            if place in self.waypoints:
                x, y = self.waypoints[place]
                self.goal = Goal(name=place, x=x, y=y, active=True)
                return f"Okay. Heading to {place}."
            else:
                return f"I don't know '{place}' yet. Add it to {config.WAYPOINTS_PATH}."

        # "set waypoint kitchen at x y"
        m = re.search(r"set waypoint\s+(.+?)\s+at\s+(-?\d+(\.\d+)?)\s+(-?\d+(\.\d+)?)", t)
        if m:
            name = m.group(1).strip().lower()
            x = float(m.group(2))
            y = float(m.group(4))
            self.waypoints[name] = (x, y)
            self._save_waypoints()
            return f"Saved waypoint {name}."

        return None

    def _save_waypoints(self) -> None:
        try:
            data = {k: {"x": v[0], "y": v[1]} for k, v in sorted(self.waypoints.items())}
            open(config.WAYPOINTS_PATH, "w", encoding="utf-8").write(json.dumps(data, indent=2))
        except Exception:
            pass

    # ---- learning updates ----
    def _learn_from_observations(self, obstacles: List[Obstacle]) -> None:
        """
        Convert perceived obstacles into experience-map updates.
        If no pose is available, we still keep a small "relative" memory by
        updating at the current pose (0,0). Once you add odometry, this becomes real.
        """
        pose = self._pose if self._has_pose else Pose2D()
        for o in obstacles:
            # only store nearby/high-confidence hazards so the map doesn't fill with noise
            if o.distance > 6.0 or o.confidence < 0.35:
                continue
            sev = 1.0
            if o.kind in ("pothole", "hole", "crack", "stairs", "step", "curb"):
                sev = 1.8
            if o.kind == "person":
                sev = 1.0
            self.exp.observe_hazard(pose, o.bearing, o.distance, kind=o.kind, severity=sev)

    # ---- action selection ----
    def _compute_action(self, obstacles: List[Obstacle]) -> Tuple[Twist2D, Optional[str]]:
        # immediate safety
        danger = self._closest_in_front(obstacles, cone_deg=25)
        if danger and danger.distance < config.STOP_DIST_M:
            return Twist2D(0.0, 0.0), f"Obstacle ahead. Stopping."

        is_crowd, crowd_dist = detect_crowd(obstacles)
        if is_crowd:
            # decide a side (simple social navigation): turn toward the side with fewer people
            left = sum(1 for o in obstacles if o.kind == "person" and o.bearing > 0)
            right = sum(1 for o in obstacles if o.kind == "person" and o.bearing < 0)
            self._prefer_left = (left < right) if (left != right) else self._prefer_left

        # choose goal bearing
        goal_bearing, goal_dist = self._goal_vector()

        # candidate actions
        fwd_set = [0.0, 0.25, 0.45, 0.65]
        turn_set = [-0.85, -0.55, -0.30, -0.15, 0.0, 0.15, 0.30, 0.55, 0.85]

        best = (-1e9, Twist2D(0.0, 0.0))
        pose = self._pose if self._has_pose else Pose2D()

        for fwd in fwd_set:
            for turn in turn_set:
                tw = Twist2D(fwd=fwd, turn=turn)
                score = self._score_action(tw, obstacles, pose, goal_bearing, goal_dist, is_crowd)
                if score > best[0]:
                    best = (score, tw)

        tw = best[1]

        # slow down near obstacles even if not stopping
        if danger and danger.distance < config.SLOW_DIST_M:
            tw.fwd = min(tw.fwd, 0.25)

        # if no goal, default to safe "follow/open space" behavior
        if not self.goal.active:
            # move forward gently if clear, else rotate to search for clear direction
            if danger and danger.distance < 1.2:
                return Twist2D(0.0, 0.45 if self._prefer_left else -0.45), "Looking for a clear path."
            return Twist2D(0.35, 0.0), None

        # reached goal?
        if self._has_pose and goal_dist < 0.7:
            self.goal.active = False
            return Twist2D(0.0, 0.0), f"Arrived at {self.goal.name}."

        return tw, None

    def _closest_in_front(self, obstacles: List[Obstacle], cone_deg: float = 25) -> Optional[Obstacle]:
        cone = math.radians(cone_deg)
        in_cone = [o for o in obstacles if abs(o.bearing) <= cone]
        if not in_cone:
            return None
        return min(in_cone, key=lambda o: o.distance)

    def _goal_vector(self) -> Tuple[float, float]:
        """
        Returns (bearing_rad, distance_m) toward the active goal.
        If no pose, returns a neutral forward goal.
        """
        if not self.goal.active:
            return (0.0, 999.0)
        if not self._has_pose:
            return (0.0, 999.0)

        dx = self.goal.x - self._pose.x
        dy = self.goal.y - self._pose.y
        dist = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        bearing = _wrap_pi(ang - self._pose.yaw)
        return (bearing, dist)

    def _score_action(self, tw: Twist2D, obstacles: List[Obstacle], pose: Pose2D,
                      goal_bearing: float, goal_dist: float, is_crowd: bool) -> float:
        """
        Higher score is better.
        """
        # Predict a short segment ahead in the direction of the candidate turn.
        # We don't simulate dynamics; we just evaluate "if I head this way, is it safe/fast?"
        look_bearing = _wrap_pi(goal_bearing + tw.turn * 0.55)  # turn changes effective heading
        look_dist = 1.5 + 1.5 * tw.fwd

        # 1) Goal alignment (prefer moving toward goal)
        goal_align = math.cos(look_bearing)  # 1 when straight to goal
        goal_term = (0.9 * goal_align + 0.4) * tw.fwd  # encourage forward motion when aligned

        # 2) Obstacle repulsion
        obs_term = 0.0
        for o in obstacles:
            # treat obstacles more strongly if they're in the direction we plan to go
            ang = abs(_wrap_pi(o.bearing - look_bearing))
            if ang > math.radians(60):
                continue
            w_ang = math.cos(ang) ** 2
            # inverse-square repulsion
            rep = w_ang * (1.0 / max(0.35, o.distance) ** 2)
            # people repulsion is softer unless it is a crowd
            if o.kind == "person" and not is_crowd:
                rep *= 0.35
            obs_term -= 2.5 * rep

        # 3) Learned cost penalty
        learned = self.exp.path_cost(pose, bearing=look_bearing, distance=look_dist)
        learned_term = -config.MAP_PATH_PENALTY * (learned - 1.0)

        # 4) Smoothness: penalize spinning
        smooth = -0.18 * abs(tw.turn)

        # 5) Crowd side bias
        crowd_bias = 0.0
        if is_crowd:
            crowd_bias = 0.12 if (tw.turn > 0 and self._prefer_left) else 0.12 if (tw.turn < 0 and not self._prefer_left) else -0.05

        return goal_term + obs_term + learned_term + smooth + crowd_bias
