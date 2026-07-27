# -*- coding: utf-8 -*-
"""BrepMFR semantic/instance segmentation training entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import time
from typing import Optional, Sequence

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # Parser/configuration inspection remains import-safe.
    pl = None


CADSynth = None
BrepSeg = None
ModelCheckpoint = None
TensorBoardLogger = None
TextMetricsLogger = None
PreparationConfig = None
ensure_prepared_dataset = None

EXPERIMENT_ALIASES = (
    "baseline",
    "topo_aag",
    "uv_aag",
    "pc_aag",
    "uv_pc_aag",
    "dynamic_no_loss",
    "affinity_only",
    "instance",
    "oracle",
    "cyl",
    "cyl_affinity",
    "point",
    "spe_pcaag",
    "full",
)


@dataclass(frozen=True)
class ExperimentSwitches:
    experiment_mode: str
    use_face_uv: bool
    use_face_pointcloud: bool
    use_cylindrical_pos: bool
    instance_bias_scale: float


_EXPERIMENT_SWITCHES = {
    "baseline": ExperimentSwitches("baseline", True, False, False, 1.0),
    "topo_aag": ExperimentSwitches("baseline", False, False, False, 1.0),
    "uv_aag": ExperimentSwitches("baseline", True, False, False, 1.0),
    "pc_aag": ExperimentSwitches("baseline", False, True, False, 1.0),
    "uv_pc_aag": ExperimentSwitches("baseline", True, True, False, 1.0),
    "dynamic_no_loss": ExperimentSwitches(
        "dynamic", True, False, False, 1.0
    ),
    "affinity_only": ExperimentSwitches(
        "dynamic_affinity", True, False, False, 0.0
    ),
    "instance": ExperimentSwitches(
        "dynamic_affinity", True, False, False, 1.0
    ),
    "oracle": ExperimentSwitches("oracle", True, False, False, 1.0),
    "cyl": ExperimentSwitches("baseline", True, False, True, 1.0),
    "cyl_affinity": ExperimentSwitches(
        "dynamic_affinity", True, False, True, 0.0
    ),
    "point": ExperimentSwitches("baseline", True, True, False, 1.0),
    "spe_pcaag": ExperimentSwitches("baseline", True, True, True, 1.0),
    "full": ExperimentSwitches(
        "dynamic_affinity", True, True, True, 1.0
    ),
}


def resolve_experiment_alias(alias: str) -> ExperimentSwitches:
    """Resolve one formal ablation name to its model-consumption switches."""
    try:
        return _EXPERIMENT_SWITCHES[alias]
    except KeyError as error:
        raise ValueError("Unsupported experiment alias: {0}".format(alias)) from error


def _add_trainer_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if pl is not None and hasattr(pl.Trainer, "add_argparse_args"):
        return pl.Trainer.add_argparse_args(parser)
    # These preserve parser/config smoke tests when Lightning is not installed.
    # A real run loads Lightning in ``main`` before Trainer construction.
    parser.add_argument("--max_epochs", type=int, default=1000)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("BrepMFR Network model")
    parser.add_argument(
        "traintest", choices=("train", "test"), help="Whether to train or test"
    )
    parser.add_argument("--num_classes", type=int, default=25)
    parser.add_argument(
        "--dataset", choices=("cadsynth", "transfer"), default="cadsynth"
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        help="Legacy path to an already converted BIN dataset",
    )
    parser.add_argument("--raw_dataset_path", type=Path)
    parser.add_argument("--prepared_dataset_path", type=Path)
    parser.add_argument("--auto_prepare", action="store_true")
    parser.add_argument("--rebuild_prepared_data", action="store_true")
    parser.add_argument("--sample_percent", type=float, default=0.01)
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--extractor_file", type=Path)
    parser.add_argument("--schema_file", type=Path)
    parser.add_argument(
        "--use_face_uv",
        dest="use_face_uv",
        action="store_true",
        help="Use the per-face UV grid encoder (enabled by default)",
    )
    parser.add_argument(
        "--no_face_uv",
        dest="use_face_uv",
        action="store_false",
        help="Disable the per-face UV grid encoder for Topo/PC-AAG ablations",
    )
    parser.set_defaults(use_face_uv=True)
    parser.add_argument("--use_face_pointcloud", action="store_true")
    parser.add_argument("--use_cylindrical_pos", action="store_true")
    parser.add_argument("--pointcloud_scale", type=float, default=1.0)
    parser.add_argument("--cylindrical_scale", type=float, default=1.0)
    parser.add_argument(
        "--point_sampling_mode",
        choices=("fixed", "area_adaptive"),
        default="area_adaptive",
    )
    parser.add_argument("--fixed_point_count", type=int, default=64)
    parser.add_argument("--min_point_count", type=int, default=32)
    parser.add_argument("--max_point_count", type=int, default=96)
    parser.add_argument("--area_sampling_exponent", type=float, default=0.5)
    parser.add_argument(
        "--candidate_grids",
        type=int,
        nargs="+",
        default=[20, 32, 48, 64, 80],
    )
    parser.add_argument("--random_rotate", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12,
        help="Use 0 on Windows; worker processes otherwise perform poorly",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--experiment_name", default="BrepMFR")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--act-dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_node", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers_encode", type=int, default=8)
    parser.add_argument(
        "--experiment_alias", choices=EXPERIMENT_ALIASES, default=None
    )
    parser.add_argument(
        "--experiment_mode",
        choices=("baseline", "oracle", "dynamic", "dynamic_affinity"),
        default="baseline",
        help="Direct instance-bias mode; retained for oracle/dynamic experiments",
    )
    parser.add_argument("--instance_bias_scale", type=float, default=1.0)
    parser.add_argument("--affinity_dim", type=int, default=64)
    parser.add_argument("--affinity_temperature", type=float, default=0.2)
    parser.add_argument("--affinity_loss_weight", type=float, default=0.1)
    parser.add_argument("--affinity_warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    return _add_trainer_arguments(parser)


def seed_everything(seed: int) -> None:
    """Seed the single project-wide reproducibility implementation."""
    from utils.reproducibility import seed_everything as seed_project

    seed_project(seed)


def _load_preparation_dependencies() -> None:
    global PreparationConfig, ensure_prepared_dataset
    if PreparationConfig is not None and ensure_prepared_dataset is not None:
        return
    from tools.prepare_dataset import (
        PreparationConfig as config_type,
        ensure_prepared_dataset as prepare,
    )

    PreparationConfig = config_type
    ensure_prepared_dataset = prepare


def _load_training_dependencies() -> None:
    global pl, CADSynth, BrepSeg
    global ModelCheckpoint, TensorBoardLogger, TextMetricsLogger
    if pl is None:
        import pytorch_lightning as lightning

        pl = lightning
    if CADSynth is None:
        from data.dataset import CADSynth as dataset_type

        CADSynth = dataset_type
    if BrepSeg is None:
        from models.brepseg_model import BrepSeg as model_type

        BrepSeg = model_type
    if ModelCheckpoint is None:
        from pytorch_lightning.callbacks import ModelCheckpoint as checkpoint_type

        ModelCheckpoint = checkpoint_type
    if TensorBoardLogger is None:
        from pytorch_lightning.loggers import TensorBoardLogger as logger_type

        TensorBoardLogger = logger_type
    if TextMetricsLogger is None:
        from training_callbacks import TextMetricsLogger as text_logger_type

        TextMetricsLogger = text_logger_type


def _manifest_count(dataset_root: Path, split: str) -> Optional[int]:
    manifest = dataset_root / (split + ".txt")
    if not manifest.is_file():
        return None
    return sum(
        1
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def prepare_training_data(args: argparse.Namespace) -> Path:
    """Prepare/reuse the new BIN contract or resolve the legacy BIN root."""
    if not args.auto_prepare:
        if args.dataset_path is None:
            raise ValueError(
                "--dataset_path is required when --auto_prepare is absent"
            )
        dataset_root = Path(args.dataset_path)
        args.preparation_reuse_status = "legacy"
        args.selected_split_counts = {
            split: _manifest_count(dataset_root, split)
            for split in ("train", "val", "test")
        }
        return dataset_root

    missing = [
        option
        for option in (
            "raw_dataset_path",
            "prepared_dataset_path",
            "extractor_file",
            "schema_file",
        )
        if getattr(args, option) is None
    ]
    if missing:
        raise ValueError(
            "--auto_prepare requires {0}".format(
                ", ".join("--" + option for option in missing)
            )
        )

    _load_preparation_dependencies()
    config = PreparationConfig(
        raw_root=args.raw_dataset_path,
        prepared_root=args.prepared_dataset_path,
        extractor_file=args.extractor_file,
        schema_file=args.schema_file,
        seed=args.seed,
        split_ratios=tuple(args.split_ratios),
        sample_percent=args.sample_percent,
        point_sampling_mode=args.point_sampling_mode,
        fixed_point_count=args.fixed_point_count,
        min_point_count=args.min_point_count,
        max_point_count=args.max_point_count,
        area_sampling_exponent=args.area_sampling_exponent,
        candidate_grids=tuple(args.candidate_grids),
        # Cache identity is deliberately independent of model/alias switches.
        face_pointcloud=True,
        cylindrical_pos=True,
        rebuild=args.rebuild_prepared_data,
    )
    prepared = ensure_prepared_dataset(config)
    args.preparation_reuse_status = "reused" if prepared.reused else "built"
    args.selected_split_counts = {
        "train": len(prepared.train_ids),
        "val": len(prepared.val_ids),
        "test": len(prepared.test_ids),
    }
    return Path(prepared.root)


def _apply_alias(args: argparse.Namespace) -> None:
    if args.experiment_alias is None:
        return
    switches = resolve_experiment_alias(args.experiment_alias)
    args.experiment_mode = switches.experiment_mode
    args.use_face_uv = switches.use_face_uv
    args.use_face_pointcloud = switches.use_face_pointcloud
    args.use_cylindrical_pos = switches.use_cylindrical_pos
    args.instance_bias_scale = switches.instance_bias_scale


def _validate_configuration(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.random_rotate and args.use_cylindrical_pos:
        parser.error(
            "--random_rotate cannot be combined with cylindrical positions"
        )
    if args.auto_prepare:
        missing = [
            option
            for option in (
                "raw_dataset_path",
                "prepared_dataset_path",
                "extractor_file",
                "schema_file",
            )
            if getattr(args, option) is None
        ]
        if missing:
            parser.error(
                "--auto_prepare requires {0}".format(
                    ", ".join("--" + option for option in missing)
                )
            )
    elif args.dataset_path is None:
        parser.error("--dataset_path is required without --auto_prepare")
    if args.traintest == "test" and args.checkpoint is None:
        parser.error("--checkpoint is required in test mode")


def _make_run_path(experiment_name: str):
    results_path = Path(__file__).parent / "results" / experiment_name
    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%M%S")
    run_path = results_path / month_day / hour_min_second
    run_path.mkdir(parents=True, exist_ok=True)
    return results_path, run_path, month_day, hour_min_second


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _run_configuration(
    args: argparse.Namespace, dataset_root: Path
) -> dict:
    configuration = {
        key: _json_value(value) for key, value in vars(args).items()
    }
    configuration.update(
        {
            "dataset_path": str(dataset_root),
            "raw_dataset_path": _json_value(args.raw_dataset_path),
            "prepared_dataset_path": (
                str(dataset_root)
                if args.auto_prepare
                else _json_value(args.prepared_dataset_path)
            ),
            "preparation_reuse_status": getattr(
                args, "preparation_reuse_status", "unknown"
            ),
            "base_seed": args.seed,
            "selected_split_counts": getattr(
                args,
                "selected_split_counts",
                {
                    split: _manifest_count(dataset_root, split)
                    for split in ("train", "val", "test")
                },
            ),
            "output_face_pointcloud": True if args.auto_prepare else None,
            "output_cylindrical_pos": True if args.auto_prepare else None,
        }
    )
    return configuration


def _dataset_options(
    args: argparse.Namespace, dataset_root: Path, split: str
) -> dict:
    require_instance = (
        args.experiment_mode in ("oracle", "dynamic_affinity")
        if split == "train"
        else args.experiment_mode == "oracle"
    )
    return {
        "root_dir": dataset_root,
        "split": split,
        "random_rotate": args.random_rotate if split == "train" else False,
        "num_class": args.num_classes,
        "require_instance": require_instance,
        "require_face_pointcloud": args.use_face_pointcloud,
        "require_cylindrical_pos": args.use_cylindrical_pos,
        "seed": args.seed,
    }


def _build_trainer(args, callbacks, logger):
    accepted = set(inspect.signature(pl.Trainer.__init__).parameters)
    trainer_options = {
        key: value
        for key, value in vars(args).items()
        if key in accepted and key != "self"
    }
    trainer_options.update(
        callbacks=callbacks,
        logger=logger,
        accelerator="gpu",
        devices=1,
        auto_select_gpus=True,
        gradient_clip_val=1.0,
        deterministic=True,
    )
    return pl.Trainer(**trainer_options)


def _append_best_checkpoint(log_path: Path, best_model_path: str) -> None:
    message = "best_checkpoint={0}".format(best_model_path or "not_saved")
    print(message)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_alias(args)
    _validate_configuration(parser, args)

    # Required initialization order: seed, data preparation, components, Trainer.
    seed_everything(args.seed)
    dataset_root = prepare_training_data(args)
    args.dataset_path = dataset_root

    _load_training_dependencies()
    if args.dataset != "cadsynth":
        raise ValueError("Unsupported dataset")

    results_path, run_path, month_day, hour_min_second = _make_run_path(
        args.experiment_name
    )
    configuration = _run_configuration(args, dataset_root)
    print(
        "Run configuration: {0}".format(
            json.dumps(configuration, sort_keys=True, ensure_ascii=False)
        )
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="eval_loss",
        mode="min",
        dirpath=str(run_path),
        filename="best-{epoch:03d}-{eval_loss:.6f}",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    text_metrics_callback = TextMetricsLogger(
        log_path=run_path / "log.txt",
        configuration=configuration,
    )
    logger = TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second
    )

    if args.traintest == "train":
        model = BrepSeg(args)
        model.save_hyperparameters(configuration)
        train_data = CADSynth(**_dataset_options(args, dataset_root, "train"))
        val_data = CADSynth(**_dataset_options(args, dataset_root, "val"))
        train_loader = train_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = val_data.get_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        trainer = _build_trainer(
            args, [checkpoint_callback, text_metrics_callback], logger
        )
        print(
            "Logs: {0}\nCheckpoint pattern: "
            "best-<epoch>-<eval_loss>.ckpt (monitor=eval_loss, save_top_k=1)".format(
                run_path
            )
        )
        trainer.fit(model, train_loader, val_loader)
        _append_best_checkpoint(
            run_path / "log.txt", checkpoint_callback.best_model_path
        )
        return 0

    test_data = CADSynth(**_dataset_options(args, dataset_root, "test"))
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = BrepSeg.load_from_checkpoint(str(args.checkpoint))
    trainer = _build_trainer(
        args, [checkpoint_callback, text_metrics_callback], logger
    )
    trainer.test(
        model,
        dataloaders=[test_loader],
        ckpt_path=str(args.checkpoint),
        verbose=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
