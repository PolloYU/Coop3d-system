"""Stable input/output contracts for single-agent 3D detection."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Detection3D:
    """One 3D box in the ego-vehicle LiDAR coordinate system."""

    label: str
    score: float
    box: Tuple[float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if len(self.box) != 7:
            raise ValueError("box must be (x, y, z, length, width, height, yaw)")


@dataclass
class DetectionResult:
    """Output produced for one agent and one frame."""

    scene_id: str
    frame_id: str
    agent_id: str
    detections: List[Detection3D] = field(default_factory=list)
    coordinate_frame: str = "ego_lidar"
    model_name: str = "pointpillars"
    inference_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
