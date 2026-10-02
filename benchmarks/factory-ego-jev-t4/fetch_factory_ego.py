#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

UPSTREAM_URL = "https://github.com/shure-dev/small-vlm-sop-check.git"
UPSTREAM_REF = "eb52106d5684a4e951ea4cf2430818e673d228dc"

def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/content/small-vlm-sop-check"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        run(["git", "clone", "--filter=blob:none", UPSTREAM_URL, str(repo)])
    run(["git", "checkout", UPSTREAM_REF], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "opencv-python-headless"])
    cmd = [sys.executable, "tools/benchmark/fetch_factory_ego.py"]
    if not args.dry_run:
        cmd.append("--apply")
    run(cmd, cwd=repo)

if __name__ == "__main__":
    main()
