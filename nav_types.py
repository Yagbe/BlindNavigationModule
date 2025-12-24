"""
Shared lightweight types for Beacon navigation.
These keep the learning planner independent from any ROS stack.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class Pose2D:
    """Pose in a local metric frame (meters, radians)."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0   # heading (rad)

@dataclass
class Twist2D:
    """Simple planar motion command (normalized)."""
    fwd: float = 0.0   # [-1,1] forward
    turn: float = 0.0  # [-1,1] left positive

@dataclass
class Obstacle:
    """Obstacle estimate in robot frame."""
    bearing: float      # radians, 0=straight ahead, +left
    distance: float     # meters
    kind: str = "unknown"
    confidence: float = 0.0
    severity: float = 1.0  # 1..?
