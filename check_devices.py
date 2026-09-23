"""What can this machine run? Run this first, before the app.

    python check_devices.py

Prints the PyTorch build, whether a GPU is reachable, and - when it is not - the
most likely reason. "It only shows CPU" becomes a one-command diagnosis instead
of a conversation.
"""
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
from engine import describe_environment


def main():
    print("=" * 60)
    print("  Face recognition benchmark - environment check")
    print("=" * 60)

    print(f"\n  Python    {platform.python_version()}  ({platform.machine()})")
    print(f"  OS        {platform.system()} {platform.release()}")

    info = describe_environment()
    if info["torch"] is None:
        print("\n  PyTorch   NOT INSTALLED")
        for note in info["notes"]:
            print(f"            {note}")
        raise SystemExit(1)

    print(f"  PyTorch   {info['torch']}"
          f"  (CUDA build: {info['cuda_build'] or 'none - CPU-only wheel'})")

    print("\n  Devices available to the app:")
    print("    cpu    yes")
    if "gpu" in info["devices"]:
        api = {"cuda": "CUDA", "xpu": "XPU"}[info["backend"]]
        print(f"    gpu    yes   {info['gpu_name']}  via {api}")
    else:
        print("    gpu    NO")
    for note in info["notes"]:
        print(f"\n  ! {note}")

    print("\n  Paths:")
    for label, path in (("models ", C.MODELS_DIR), ("data   ", C.DATA_DIR),
                        ("assets ", C.ASSETS_DIR)):
        mark = "" if os.path.isdir(path) else "   (does not exist yet)"
        print(f"    {label} {path}{mark}")

    weights = C.find_weights()
    print(f"\n  Checkpoints found under models: {len(weights)}")
    for w in weights[:5]:
        print(f"    {os.path.basename(w)}  ({os.path.getsize(w) / 1e6:.0f} MB)")
    if not weights:
        print("    none - you can still type a full path in the app's setup screen.")

    print()
    if "gpu" in info["devices"]:
        print("  Ready: both CPU and GPU measurements are possible.")
    else:
        print("  Ready: CPU measurements only on this machine.")
    print("  Start the app with:  python -m streamlit run app.py\n")


if __name__ == "__main__":
    main()
