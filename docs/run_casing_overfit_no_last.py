"""Run segmentation.py with best-only checkpoints for the 001 debug trial.

Pass the same arguments that segmentation.py accepts. This wrapper changes
only the checkpoint callbacks; it avoids overwriting last.ckpt each epoch.
"""

from pathlib import Path
import sys

from pytorch_lightning.callbacks import ModelCheckpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import segmentation


def build_callbacks(run_path):
    best_miou = ModelCheckpoint(
        monitor="mean_iou", mode="max", dirpath=str(run_path),
        filename="best-mIoU-{epoch:03d}-{mean_iou:.6f}",
        save_top_k=1, save_last=False,
        auto_insert_metric_name=False,
    )
    best_macro_f1 = ModelCheckpoint(
        monitor="macro_f1", mode="max", dirpath=str(run_path),
        filename="best-Macro-F1-{epoch:03d}-{macro_f1:.6f}",
        save_top_k=1, save_last=False,
        auto_insert_metric_name=False,
    )
    return best_miou, best_macro_f1


def main(argv=None):
    segmentation._build_checkpoint_callbacks = build_callbacks
    return segmentation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
