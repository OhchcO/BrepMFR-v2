#!/usr/bin/env python
"""Recognize machining features in one STEP file and write canonical JSON."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import traceback
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brepmfr.inference.pipeline import (
    InferencePipelineError,
    InferenceRequest,
    run_step_inference,
)


def _affinity_threshold(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a number") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be finite and in [0, 1]")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, type=Path, help="input STEP file")
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="BrepMFR .ckpt file"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="result and BIN cache directory"
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--affinity-threshold", type=_affinity_threshold, default=0.5
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser


def _validate_paths(parser: argparse.ArgumentParser, arguments) -> None:
    step = arguments.step.resolve()
    checkpoint = arguments.checkpoint.resolve()
    output_dir = arguments.output_dir.resolve()
    if step.suffix.lower() not in (".step", ".stp"):
        parser.error("--step must use a .step or .stp suffix")
    if not step.is_file():
        parser.error("--step does not exist: {0}".format(step))
    if checkpoint.suffix.lower() != ".ckpt":
        parser.error("--checkpoint must use the .ckpt suffix")
    if not checkpoint.is_file():
        parser.error("--checkpoint does not exist: {0}".format(checkpoint))
    if output_dir.exists() and not output_dir.is_dir():
        parser.error("--output-dir is not a directory: {0}".format(output_dir))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _validate_paths(parser, arguments)
    request = InferenceRequest(
        step_path=arguments.step,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        device=arguments.device,
        affinity_threshold=arguments.affinity_threshold,
        force_rebuild=arguments.force_rebuild,
        seed=arguments.seed,
    )
    try:
        outcome = run_step_inference(request)
    except InferencePipelineError as error:
        traceback.print_exc(file=sys.stderr)
        if error.result_path is not None:
            print("Error result: {0}".format(error.result_path), file=sys.stderr)
        return 1

    print("Result: {0}".format(outcome.result_path))
    print(
        "Faces: {0}; instances: {1}; cache: {2}".format(
            len(outcome.document["faces"]),
            len(outcome.document["instances"]),
            "hit" if outcome.cache_hit else "miss",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

