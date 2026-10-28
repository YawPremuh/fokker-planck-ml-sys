import json
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

def command(*arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "Unavailable"

def main():
    environment = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "logical_cpus": command("sysctl", "-n", "hw.logicalcpu"),
        "memory_bytes": command("sysctl", "-n", "hw.memsize"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": str(torch.__version__),
        "mps_available": torch.backends.mps.is_available(),
        "compiler": command("clang++", "--version"),
        "cmake": command("cmake", "--version"),
        "git_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "git_status": command("git", "-C", str(ROOT), "status", "--short"),
    }
    destination = ROOT / "results" / "environment.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(environment, indent=2) + "\n")
    print(f"Saved environment details: {destination}")

if __name__ == "__main__":
    main()
