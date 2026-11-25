"""Record driver capabilities without modifying device state."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import csv
import os
import platform
import subprocess


def command(*args, timeout=6):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def query(fields, uuid=None):
    args = ["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    if uuid:
        args.extend(["-i", uuid])
    result = command("nvidia-smi", *args)
    if result["returncode"] != 0:
        return []
    return [dict(zip(fields, (value.strip() for value in row)))
            for row in csv.reader(result["stdout"].splitlines())]


def inventory():
    return query(("index", "name", "uuid", "memory.total", "driver_version", "pci.bus_id",
                  "mig.mode.current"))


def sample(uuid):
    fields = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
              "utilization.gpu", "utilization.memory", "memory.used", "pstate")
    return {"gpu_uuid": uuid, "utc": datetime.now(timezone.utc).isoformat(),
            "values": next(iter(query(fields, uuid)), {})}


def a100_devices(indices):
    devices = {row["index"]: row for row in inventory()}
    if not 1 <= len(indices) <= 2 or len(set(indices)) != len(indices):
        raise ValueError("Select one or two distinct physical GPU indices")
    found = [devices.get(i, {}) for i in indices]
    for index, row in zip(indices, found):
        if "A100" not in row.get("name", ""):
            existing = ", ".join(f'{i} {d["name"]}' for i, d in devices.items()) or "none"
            raise ValueError(f"GPU {index} is not an accessible A100. Visible devices: {existing}")
        if row["mig.mode.current"] == "Enabled":
            raise ValueError(f"GPU {index} has MIG enabled. Capture requires a full physical A100")
    return found


def hardware_report():
    packages = {}
    for name in ("torch", "triton", "numpy", "scipy", "matplotlib", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "python": platform.python_version(), "packages": packages, "gpus": inventory(),
            "uid": os.geteuid(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "wsl_device": Path("/dev/dxg").exists(), "nvidia_smi": command("nvidia-smi"),
            "topology": command("nvidia-smi", "topo", "-m"),
            "cuda_toolkit": command("nvcc", "--version"),
            "supported_clocks": command("nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"),
            "clock_lock_permissions": "unverified; clock settings were not changed"}
