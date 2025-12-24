"""
Experience-based learning layer for Beacon.

What "learning" means here (safe + implementable):
- The robot maintains a small 2D grid map of "how risky/slow" each area is.
- When it *sees* hazards (potholes, crowds, etc.) it increases cost in those cells.
- When it *traverses* cells successfully it slightly reduces cost.
- Costs decay over time so old obstacles don't dominate forever.

This is not black-box end-to-end RL. It's an online, explainable learning approach
that you can validate and tune on a real assistive robot.
"""
from __future__ import annotations

import json, math, time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any

import numpy as np

import config
from nav_types import Pose2D

Cell = Tuple[int, int]

def _now() -> float:
    return time.time()

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class CellInfo:
    cost: float = 1.0
    last_update: float = 0.0
    last_seen_kind: str = ""
    seen_count: int = 0

class ExperienceMap:
    """
    Sparse grid map: only stores cells that have been updated.
    """
    def __init__(self, res_m: float = None):
        self.res_m = float(res_m if res_m else config.MAP_RES_M)
        self._cells: Dict[Cell, CellInfo] = {}
        self._loaded_path: Optional[str] = None

    # ---- persistence ----
    def load(self, path: str) -> None:
        self._loaded_path = path
        try:
            data = json.loads(open(path, "r", encoding="utf-8").read())
        except FileNotFoundError:
            return
        except Exception:
            # if corrupted, ignore rather than crashing navigation
            return

        self.res_m = float(data.get("res_m", self.res_m))
        cells = data.get("cells", {})
        self._cells.clear()
        for k, v in cells.items():
            try:
                ix, iy = k.split(",")
                cell = (int(ix), int(iy))
                info = CellInfo(
                    cost=float(v.get("cost", 1.0)),
                    last_update=float(v.get("last_update", 0.0)),
                    last_seen_kind=str(v.get("last_seen_kind", "")),
                    seen_count=int(v.get("seen_count", 0)),
                )
                self._cells[cell] = info
            except Exception:
                continue

    def save(self, path: Optional[str] = None) -> None:
        p = path or self._loaded_path
        if not p:
            return
        cells = {}
        for (ix, iy), info in self._cells.items():
            cells[f"{ix},{iy}"] = asdict(info)
        data = {"res_m": self.res_m, "cells": cells, "saved_at": _now()}
        try:
            open(p, "w", encoding="utf-8").write(json.dumps(data, indent=2))
        except Exception:
            pass

    # ---- map helpers ----
    def _decay_cell(self, info: CellInfo, t: float) -> None:
        if info.last_update <= 0:
            info.last_update = t
            return
        dt_min = max(0.0, (t - info.last_update) / 60.0)
        if dt_min <= 0:
            return
        # Exponential-ish decay to baseline cost=1
        decay = config.MAP_DECAY_PER_MIN
        # move cost toward 1.0
        info.cost = 1.0 + (info.cost - 1.0) * math.exp(-decay * dt_min)
        info.last_update = t

    def _cell_of(self, x: float, y: float) -> Cell:
        return (int(math.floor(x / self.res_m)), int(math.floor(y / self.res_m)))

    def _world_from_pose(self, pose: Pose2D, bearing: float, distance: float) -> Tuple[float, float]:
        # convert robot-frame polar (bearing, distance) into world XY using pose
        ang = pose.yaw + bearing
        return (pose.x + math.cos(ang) * distance, pose.y + math.sin(ang) * distance)

    # ---- learning updates ----
    def observe_hazard(self, pose: Pose2D, bearing: float, distance: float,
                       kind: str, severity: float = 1.0, bump: bool = False) -> None:
        """
        Update map costs based on a perceived hazard.
        """
        t = _now()
        wx, wy = self._world_from_pose(pose, bearing, distance)
        cell = self._cell_of(wx, wy)
        info = self._cells.get(cell, CellInfo(last_update=t))
        self._decay_cell(info, t)

        bonus = config.MAP_OBS_BONUS * float(severity)
        if bump:
            bonus += config.MAP_BUMP_BONUS

        info.cost = _clamp(info.cost + bonus, 1.0, 50.0)
        info.last_seen_kind = str(kind)
        info.seen_count += 1
        info.last_update = t
        self._cells[cell] = info

    def mark_traversed(self, pose: Pose2D, success: bool = True) -> None:
        """
        Called periodically as the robot moves.
        Successful traversal reduces cost slightly (learning that the area is OK).
        """
        t = _now()
        cell = self._cell_of(pose.x, pose.y)
        info = self._cells.get(cell, CellInfo(last_update=t))
        self._decay_cell(info, t)
        if success:
            info.cost = _clamp(info.cost - 0.15, 1.0, 50.0)
        else:
            info.cost = _clamp(info.cost + 0.8, 1.0, 50.0)
        info.last_update = t
        self._cells[cell] = info

    # ---- querying ----
    def cost_at(self, x: float, y: float) -> float:
        t = _now()
        cell = self._cell_of(x, y)
        info = self._cells.get(cell)
        if not info:
            return 1.0
        self._decay_cell(info, t)
        return float(info.cost)

    def path_cost(self, pose: Pose2D, bearing: float, distance: float, n: int = 6) -> float:
        """
        Approximate average learned cost along a straight segment ahead.
        """
        if distance <= 0:
            return 1.0
        total = 0.0
        for i in range(1, n + 1):
            d = distance * (i / n)
            wx, wy = self._world_from_pose(pose, bearing, d)
            total += self.cost_at(wx, wy)
        return total / n

    def snapshot(self, max_cells: int = 250) -> Dict[str, Any]:
        """
        For debugging / UI. Returns a small subset of learned cells.
        """
        items = sorted(self._cells.items(), key=lambda kv: kv[1].cost, reverse=True)[:max_cells]
        out = []
        for (ix, iy), info in items:
            out.append({"ix": ix, "iy": iy, "cost": info.cost, "kind": info.last_seen_kind, "n": info.seen_count})
        return {"res_m": self.res_m, "cells": out}
