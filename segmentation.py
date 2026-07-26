# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg
from models.modules.utils.macro import *
from training_callbacks import TextMetricsLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser("BrepMFR Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="cadsynth", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=12,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="BrepMFR",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

#设置transformer模块的默认参数
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--attention_dropout", type=float, default=0.3)
parser.add_argument("--act-dropout", type=float, default=0.3)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_node", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
parser.add_argument("--n_layers_encode", type=int, default=8)
parser.add_argument(
    "--experiment_mode",
    choices=("baseline", "oracle", "dynamic", "dynamic_affinity"),
    default="baseline",
    help="Instance-bias ablation mode",
)
parser.add_argument("--instance_bias_scale", type=float, default=1.0)
parser.add_argument("--affinity_dim", type=int, default=64)
parser.add_argument("--affinity_temperature", type=float, default=0.2)
parser.add_argument("--affinity_loss_weight", type=float, default=0.1)
parser.add_argument("--affinity_warmup_epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=2026)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
pl.seed_everything(args.seed, workers=True)
torch.backends.cudnn.benchmark = False

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
run_path = results_path.joinpath(month_day, hour_min_second)
run_path.mkdir(parents=True, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",
    dirpath=str(run_path),
    filename="best",
    save_top_k=10,
    save_last=True,
)
text_metrics_callback = TextMetricsLogger(
    log_path=run_path.joinpath("log.txt"),
    configuration=vars(args),
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback, text_metrics_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True, 
    gradient_clip_val=1.0,
    deterministic=True,
)

if args.dataset == "cadsynth":
    Dataset = CADSynth
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train":
    # Train/val
    print(
        f"""
-----------------------------------------------------------------------------------
B-rep model feature recognition
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}
Text metrics written to results/{args.experiment_name}/{month_day}/{hour_min_second}/log.txt

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = BrepSeg(args)

    train_data = Dataset(
        root_dir=args.dataset_path,
        split="train",
        random_rotate=True,
        num_class=args.num_classes,
        require_instance=args.experiment_mode in ("oracle", "dynamic_affinity"),
    )
    val_data = Dataset(
        root_dir=args.dataset_path,
        split="val",
        random_rotate=False,
        num_class=args.num_classes,
        require_instance=args.experiment_mode == "oracle",
    )
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )    
    trainer.fit(model, train_loader, val_loader)
    
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(
        root_dir=args.dataset_path,
        split="test",
        random_rotate=False,
        num_class=args.num_classes,
        require_instance=args.experiment_mode == "oracle",
    )
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = BrepSeg.load_from_checkpoint(args.checkpoint)
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)
    
