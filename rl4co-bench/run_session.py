#!/usr/bin/env python3
"""Calibrate, profile, and summarize one T2 CUDA measurement session."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def run(script: str, *args) -> None:
    command = [sys.executable, "-B", str(BENCH / script), *map(str, args)]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New session directory")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained AM policy weights")
    parser.add_argument("--checkpoint-source", required=True, help="Checkpoint origin and training size")
    parser.add_argument("--policy-kwargs", type=Path, help="Constructor options for a nondefault AM checkpoint")
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--case", action="append", default=[], help="Run only this sweep case (repeatable)")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--enrol-reference", action="store_true", help="Use this session as the CUDA calibration reference")
    selection.add_argument("--reference", type=Path, help="Compare with an earlier enrolled calibration JSON")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.is_relative_to((BENCH.parent / "rl4co").resolve()):
        parser.error("The session directory must be outside vendored RL4CO")
    if not args.checkpoint.is_file() or not args.checkpoint_source.strip():
        parser.error("Supply an existing trained checkpoint and its provenance")
    if args.policy_kwargs is not None and not args.policy_kwargs.is_file():
        parser.error("--policy-kwargs must point to a readable JSON file")
    if args.reference is not None and not args.reference.is_file():
        parser.error("--reference must point to the enrolled calibration JSON")

    try:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "pip-freeze.txt").open("x", encoding="utf-8") as versions:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=versions)
        calibration = output / "calibration.json"
        ref_args = ("--enrol-reference",) if args.enrol_reference else ("--reference", args.reference)
        run("envs/session.py", "--output", calibration, "--device", args.device, *ref_args)

        trace_args = []
        if args.policy_kwargs is not None:
            trace_args += ["--policy-kwargs", args.policy_kwargs]
        for case in args.case:
            trace_args += ["--case", case]
        profile = output / "profile"
        run(
            "profiling/run_profile.py", "--output", profile,
            "--checkpoint", args.checkpoint, "--checkpoint-source", args.checkpoint_source,
            "--calibration", calibration, "--device-index", args.device, *trace_args,
        )
        run("profiling/summarize.py", profile, "--output", output / "report")
    except subprocess.CalledProcessError as exc:
        parser.exit(exc.returncode or 1, f"Session stopped at {Path(exc.cmd[2]).name}. See {output}\n")
    except OSError as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
