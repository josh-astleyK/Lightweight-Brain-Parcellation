"""Patch mamba-ssm for CPU-only inference (Linux CPU or Windows).

mamba-ssm's package __init__ imports Mamba2 and mixer_seq_simple, which pull in
triton (Linux/CUDA only). This script rewrites __init__ to expose only the
Mamba1 block used by these models, and writes stub modules for the compiled CUDA
extensions so the import succeeds without a GPU build.

Run once after installing mamba-ssm on a CPU-only machine:
    python fix_mamba_cpu.py
"""
import importlib.util
import os
import sys

MARKER = "# CPU-ONLY PATCH"
PATCHED_INIT = (
    MARKER + "\n"
    "from mamba_ssm.modules.mamba_simple import Mamba\n"
)

STUB = (
    '"""CPU stub — imported at load time but never called during CPU inference."""\n'
    "def __getattr__(attr):\n"
    "    raise RuntimeError(\n"
    '        f"{__name__}.{attr} called: this is a CPU stub. '
    'The CUDA fast path must be disabled."\n'
    "    )\n"
)


def _patch_init(pkg_dir):
    path = os.path.join(pkg_dir, "__init__.py")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        print(f"[fix] cannot read {path}: {exc}")
        return
    if MARKER in text:
        print("[fix] __init__.py already patched.")
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        open(backup, "w", encoding="utf-8").write(text)
    open(path, "w", encoding="utf-8").write(PATCHED_INIT)
    print(f"[fix] patched {path}")


def _write_stub(site_packages, name):
    path = os.path.join(site_packages, name + ".py")
    if os.path.exists(path):
        print(f"[fix] stub {name} already present.")
        return
    open(path, "w", encoding="utf-8").write(STUB)
    print(f"[fix] wrote stub for {name}")


def main():
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        print("[fix] mamba_ssm not found; install it first.")
        return 1

    pkg_dir = list(spec.submodule_search_locations)[0]
    site_packages = os.path.dirname(pkg_dir)

    _patch_init(pkg_dir)
    _write_stub(site_packages, "selective_scan_cuda")
    _write_stub(site_packages, "causal_conv1d_cuda")

    result = __import__("subprocess").run(
        [sys.executable, "-c", "import mamba_ssm; print(mamba_ssm.Mamba)"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[fix] OK — {result.stdout.strip()}")
    else:
        print("[fix] import still failing:")
        sys.stderr.write(result.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
