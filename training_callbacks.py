"""Training callbacks shared by BrepMFR experiment entry points."""

from datetime import datetime
import json
from pathlib import Path

import torch
from pytorch_lightning import Callback


class TextMetricsLogger(Callback):
    """Write one readable training/validation record per epoch."""

    def __init__(self, log_path, configuration):
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.configuration = dict(configuration)
        with self.log_path.open("w", encoding="utf-8") as stream:
            stream.write("BrepMFR experiment log\n")
            stream.write(
                "created_at={}\n".format(
                    datetime.now().isoformat(timespec="seconds")
                )
            )
            stream.write(
                "configuration={}\n".format(
                    json.dumps(
                        self.configuration,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
            )

    def _append(self, message):
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(message.rstrip() + "\n")

    @staticmethod
    def _scalar(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def on_fit_start(self, trainer, pl_module):
        self._append(
            "[{}] fit_start device={} mode={}".format(
                datetime.now().isoformat(timespec="seconds"),
                pl_module.device,
                getattr(pl_module, "experiment_mode", "unknown"),
            )
        )

    @staticmethod
    def _available_metrics(trainer):
        metrics = dict(trainer.callback_metrics)
        metrics.update(trainer.logged_metrics)
        return metrics

    def _format_metrics(self, trainer, phase, metric_names):
        available_metrics = self._available_metrics(trainer)
        fields = [
            "[{}]".format(datetime.now().isoformat(timespec="seconds")),
            "phase={}".format(phase),
            "epoch={}".format(trainer.current_epoch + 1),
            "global_step={}".format(trainer.global_step),
        ]
        for name in metric_names:
            if name not in available_metrics:
                continue
            scalar = self._scalar(available_metrics[name])
            if scalar is not None:
                fields.append("{}={:.8f}".format(name, scalar))
        return " ".join(fields)

    def on_train_epoch_end(self, trainer, pl_module):
        self._append(
            self._format_metrics(
                trainer,
                "train",
                (
                    "current_lr",
                    "train_loss",
                    "train_semantic_loss",
                    "train_affinity_loss",
                    "train_affinity_weight",
                    "train_instance_bias_progress",
                    "train_instance_bias_gate",
                ),
            )
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        available_metrics = self._available_metrics(trainer)
        metric_names = [
            "eval_loss",
            "per_face_accuracy",
            "mean_class_accuracy",
            "mean_iou",
            "macro_f1",
        ]
        metric_names.extend(
            sorted(
                name
                for name in available_metrics
                if name.startswith("iou_class_")
            )
        )
        self._append(
            self._format_metrics(
                trainer,
                "validation",
                metric_names,
            )
        )

    def on_fit_end(self, trainer, pl_module):
        self._append(
            "[{}] fit_end epoch={} global_step={}".format(
                datetime.now().isoformat(timespec="seconds"),
                trainer.current_epoch,
                trainer.global_step,
            )
        )

    def on_exception(self, trainer, pl_module, exception):
        self._append(
            "[{}] exception type={} message={}".format(
                datetime.now().isoformat(timespec="seconds"),
                type(exception).__name__,
                str(exception).replace("\n", " "),
            )
        )
