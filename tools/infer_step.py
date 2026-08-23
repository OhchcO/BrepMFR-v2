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
from brepmfr.inference.result import build_error_document, write_json_atomic


class _CLIArgumentError(ValueError):
    """Argument parsing or validation failed."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CLIArgumentError(message)


def _affinity_threshold(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be a number") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be finite and in [0, 1]")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
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


def _option_value(argv: Sequence[str], option: str) -> Optional[str]:
    value = None
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            candidate = argv[index + 1]
            if not candidate.startswith("--"):
                value = candidate
        elif item.startswith(option + "="):
            value = item.split("=", 1)[1]
    return value


def _write_argument_error(
    argv: Sequence[str], error: BaseException
) -> Optional[Path]:
    output_value = _option_value(argv, "--output-dir")
    if not output_value:
        return None

    try:
        output_dir = Path(output_value).resolve()
        if output_dir.exists() and not output_dir.is_dir():
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        step_value = _option_value(argv, "--step")
        step_path = (
            Path(step_value).resolve()
            if step_value
            else Path.cwd() / "__missing_step_argument__.step"
        )
        result_path = output_dir / "result.json"
        document = build_error_document(
            "arguments", ValueError(str(error)), step_path
        )
        write_json_atomic(result_path, document)
        return result_path
    except (OSError, ValueError) as write_error:
        print(
            "Could not write argument error result: {0}".format(write_error),
            file=sys.stderr,
        )
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    try:
        arguments = parser.parse_args(raw_argv)
        _validate_paths(parser, arguments)
    except _CLIArgumentError as error:
        parser.print_usage(sys.stderr)
        print("{0}: error: {1}".format(parser.prog, error), file=sys.stderr)
        result_path = _write_argument_error(raw_argv, error)
        if result_path is not None:
            print("Error result: {0}".format(result_path), file=sys.stderr)
        raise SystemExit(2)
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
