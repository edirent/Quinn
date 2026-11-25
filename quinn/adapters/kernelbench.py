"""Load trusted source from the exact inspected KernelBench Git object."""
import subprocess
import sys
from types import ModuleType

REVISION = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
FP32_TOLERANCE = 1e-4


def git(root, *args):
    try:
        return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"KernelBench source unavailable. Run make fetch-upstream. {exc.stderr.decode().strip()}") from exc


def files(root):
    data = git(root, "ls-tree", "-r", "--name-only", REVISION, "--", "KernelBench")
    return data.decode().splitlines()


def source(root, path):
    return git(root, "show", f"{REVISION}:{path}")


def load(root, path, name):
    module = ModuleType(name)
    module.__file__ = str(root / path)
    sys.modules[name] = module
    exec(compile(source(root, path), module.__file__, "exec"), module.__dict__)
    return module
