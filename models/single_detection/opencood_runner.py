"""Small adapter around OpenCOOD's official inference command."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import List


@dataclass(frozen=True)
class OpenCoodConfig:
    opencood_root: Path
    model_dir: Path
    python_executable: Path = Path(sys.executable)
    fusion_method: str = "late"
    save_npy: bool = True


class OpenCoodRunner:
    """Validate and launch an external OpenCOOD installation."""

    def __init__(self, config: OpenCoodConfig) -> None:
        self.config = config

    @property
    def inference_script(self) -> Path:
        return self.config.opencood_root / "opencood" / "tools" / "inference.py"

    def problems(self) -> List[str]:
        problems = []  # type: List[str]
        if not self.config.python_executable.is_file():
            problems.append(f"Python executable not found: {self.config.python_executable}")
        if not self.inference_script.is_file():
            problems.append(f"OpenCOOD inference script not found: {self.inference_script}")
        if not self.config.model_dir.is_dir():
            problems.append(f"Model directory not found: {self.config.model_dir}")
        elif not (self.config.model_dir / "config.yaml").is_file():
            problems.append(f"Model config not found: {self.config.model_dir / 'config.yaml'}")
        elif not any(self.config.model_dir.glob("*.pth")):
            problems.append(f"No .pth checkpoint found in: {self.config.model_dir}")
        return problems

    def build_command(self) -> List[str]:
        if self.config.fusion_method not in {"late", "early", "intermediate"}:
            raise ValueError("fusion_method must be late, early, or intermediate")
        command = [
            str(self.config.python_executable),
            str(self.inference_script),
            "--model_dir",
            str(self.config.model_dir),
            "--fusion_method",
            self.config.fusion_method,
        ]
        if self.config.save_npy:
            command.append("--save_npy")
        return command

    def run(self) -> subprocess.CompletedProcess:
        problems = self.problems()
        if problems:
            raise RuntimeError("\n".join(problems))
        return subprocess.run(
            self.build_command(),
            cwd=self.config.opencood_root,
            check=True,
            text=True,
        )
