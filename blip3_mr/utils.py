import json
import os
import random
import re
import shutil
from datetime import datetime
from enum import Enum
from typing import Any

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from peft import PeftMixedModel, PeftModel
from safetensors import safe_open
from tqdm import tqdm

from blip3_mr.config import Config
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver


def isdir(path: Any):
    return path is not None and isinstance(path, str) and os.path.isdir(path)


def isfile(path: Any):
    return path is not None and isinstance(path, str) and os.path.isfile(path)


def list_dict_to_jsonl(l: list[dict]):
    return "\n".join(json.dumps(l[i]) for i in range(len(l)))


def log(message: str, **kwargs):
    """
    Logs message and timestamp only when it's rank 0

    Args:
        message (str): The message to log.
        **kwargs: Additional keyword arguments to pass to the `print` function.
    """
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(f"\n[{timestamp}] {message}", **kwargs)


def calculate_loss_weight_inner(config: Config, epoch: int = 0):
    if config is None or not os.path.isfile(config.loss_mapping_path):
        return

    with open(config.loss_mapping_path, "r") as f:
        loss_mapping = yaml.safe_load(f)

    assert isinstance(loss_mapping, dict)

    keys = list(loss_mapping.keys())

    if str(epoch) in keys:
        loss_weights = loss_mapping[str(epoch)]
    else:
        max_key = sorted(keys, reverse=True, key=lambda k: int(k))[0]
        loss_weights = loss_mapping[max_key]

    if len(loss_weights) != 4:
        return

    loss_sum = sum(loss_weights)
    loss_weights = [l / loss_sum for l in loss_weights]

    config.loss_ce_weight = loss_weights[0]
    config.loss_bce_weight = loss_weights[1]
    config.loss_tvl_weight = loss_weights[2]
    config.loss_gd_weight = loss_weights[3]


def calculate_loss_weight(config: Config, epoch: int = 0):
    """
    Calculates the loss weight for current epoch, if loss_mapping_path is a valid path with valid yaml structure.
    """
    calculate_loss_weight_inner(config, epoch)
    log("Loss Weights: ")
    if config.rank == 0:
        print(
            f"  Cross Entropy Loss Weight: {config.loss_ce_weight}\n"
            f"  Binary Cross Entropy Loss Weight: {config.loss_bce_weight}\n"
            f"  Tversky Loss Weight: {config.loss_tvl_weight}\n"
            f"  Generalized Dice Loss Weight: {config.loss_gd_weight}\n"
        )
    return dict(
        loss_ce_weight=config.loss_ce_weight,
        loss_bce_weight=config.loss_bce_weight,
        loss_tvl_weight=config.loss_tvl_weight,
        loss_gd_weight=config.loss_gd_weight,
    )


def json_dumps(data: Any, indent: int = 2, max_inline_length: int = 120) -> str:
    """
    Dumps data to a JSON string with special formatting for short lists.

    Args:
        data (any): The Python object to serialize.
        indent (int, optional): The indentation level for the JSON output. Defaults to 2.
        max_inline_length (int, optional): The maximum length for a list to be
            inlined. Defaults to 120.

    Returns:
        (str | dict): The formatted JSON string or the original data if failed.
    """
    json_str = json.dumps(data, indent=indent)
    # regex to find lists with no nested braces/brackets
    pattern = re.compile(r"\[\s*([^\[\]\{\}]+?)\s*\]", re.DOTALL)

    def replacer(match):
        content = match.group(1)
        # remove whitespace and newlines inside the list
        inline = " ".join(content.split())
        if len(inline) <= max_inline_length:
            return f"[ {inline} ]"
        else:
            return match.group(0)

    out = pattern.sub(replacer, json_str)
    return out


def random_seed(seed: int = 42):
    """
    Sets the seed for random number generators in PyTorch, NumPy, and Python's random module.
    """
    rank = int(os.environ.get("RANK", 0))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def init_wandb(config: Config):
    """
    Initializes a Weights & Biases (wandb) run based on command-line arguments. Only on rank 0.

    Args:
        config (Config): An object containing the configuration arguments for the training run.
    """
    if config.rank == 0 and config.report_to_wandb:
        wandb_config = {
            "project": config.wandb_project or "blip3-mr",
            "name": f"{config.run_name}",
            "config": vars(config),
            "entity": config.wandb_entity,
        }

        wandb.init(**wandb_config)

        if wandb.run is not None:
            for key, value in vars(config).items():
                wandb.run.config[key] = value


def unwrap_model(model: nn.Module, verbose=False) -> XGenMMPerceiver:
    """
    Recursively unwraps a model from common wrappers like DeepSpeed, DDP, and PEFT.

    Args:
        model (nn.Module): The model
        verbose (bool): verbosity

    Returns:
        output (XGenMMPerceiver): The base, unwrapped model.
    """
    wrapping_layer_names: list[str] = []
    unwrapped_model: torch.nn.Module = model
    i = -1
    while True and i < 10:
        i += 1
        if hasattr(unwrapped_model, "module") and "deepspeed" in str(type(unwrapped_model)).lower():
            wrapping_layer_names.append("DeepSpeed")
            unwrapped_model = unwrapped_model.module
            continue

        if isinstance(unwrapped_model, nn.parallel.DistributedDataParallel):
            wrapping_layer_names.append("DDP")
            unwrapped_model = unwrapped_model.module
            continue

        if isinstance(unwrapped_model, (PeftMixedModel, PeftModel)):
            wrapping_layer_names.append("PEFT")
            unwrapped_model = unwrapped_model.base_model.model
            continue

        break

    if verbose:
        print("Model wrapping layers (outer to inner):", ", ".join(wrapping_layer_names))
    return unwrapped_model


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """
    A utility class to compute and store the average and current value of a metric.
    """

    def __init__(self, name: str, fmt: str = ":f", summary_type: Summary = Summary.AVERAGE, suffix=""):
        """
        Initializes the AverageMeter.

        Args:
            name (str): The name of the metric (e.g., "Loss", "Accuracy").
            fmt (str, optional): The format specifier for printing the value. Defaults to ":f".
            summary_type (Summary, optional): The type of summary to produce.
                Defaults to Summary.AVERAGE.
        """
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0
        self.suffix: str = suffix
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int, optional): The number of samples associated with the value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """
        Synchronizes the meter's sum and count across all processes in a distributed setting.
        """
        device = torch.device("cuda")
        total = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        """Returns a string representation of the meter's current and average values."""
        if "token" in self.name or "time" in self.name or self.name == "lr":
            fmtstr = "{name} {val" + self.fmt + "}" + self.suffix
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})" + self.suffix
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        """Returns a string summarizing the meter based on its summary_type"""
        fmtstr = "\t"
        if self.summary_type is Summary.AVERAGE:
            fmtstr += "{name} avg  : {avg" + self.fmt + "}\n"
        elif self.summary_type is Summary.SUM:
            fmtstr += "{name} sum  : {sum" + self.fmt + "}\n"
        elif self.summary_type is Summary.COUNT:
            fmtstr += "{name} count: {count" + self.fmt + "}\n"
        else:
            fmtstr += "\n"

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    A utility class to display the progress of training or evaluation.
    """

    def __init__(self, num_batches: int, meters: list[AverageMeter], prefix: str = "", use_tqdm: bool = True):
        """
        Initializes the ProgressMeter.

        Args:
            num_batches (int): The total number of batches in the epoch.
            meters (list[AverageMeter]): A list of AverageMeter objects to track.
            prefix (str, optional): A prefix string for the display (e.g., "Epoch: [1]").
                Defaults to "".
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters: list[AverageMeter] = meters
        self.prefix = prefix
        self.print = tqdm.write if use_tqdm else print

    def display(self, batch: int):
        """
        Displays the current progress for a given batch.

        Args:
            batch (int): The current batch number.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.print("    ".join(entries))

    def display_summary(self):
        """Displays a summary of all meters at the end of an epoch."""
        if len(self.prefix) > 0:
            entries = [f"{self.prefix}\n"]
        else:
            entries = ["Averages:\n"]
        entries += [meter.summary() for meter in self.meters]
        self.print("    ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """
        Creates a format string for the batch number, e.g., "[  1/100]".

        Args:
            num_batches (int): The total number of batches.

        Returns:
            str: The format string.
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class TrainingMeters:
    """
    A dedicated class to hold and manage all AverageMeter instances for training.
    """

    def __init__(self, config: Config):
        """
        Initializes all the meters required for training from Config.
        """
        self.config = config
        self.should_reduce = config.world_size > 1
        # Core meters
        self.lr = AverageMeter("lr", ":.2e")
        self.step_time = AverageMeter("step_time_s", ":.1f")
        self.data_time = AverageMeter("data_time_ms", ":.1f")
        self.num_tokens = AverageMeter("tokens_k", ":.2f")
        self.num_pos = AverageMeter("positive_ratio", ":.2f")

        self.bce_loss: AverageMeter | None = None
        self.ce_loss: AverageMeter | None = None
        self.gd_loss: AverageMeter | None = None
        self.tv_loss: AverageMeter | None = None
        self.loss: AverageMeter = AverageMeter("loss", ":.4f")

        if self.config.loss_bce_weight > 0.0:
            self.bce_loss = AverageMeter("bce_loss", ":.4f")
        if self.config.loss_ce_weight > 0.0:
            self.ce_loss = AverageMeter("ce_loss", ":.4f")
        if self.config.loss_gd_weight > 0.0:
            self.gd_loss = AverageMeter("gd_loss", ":.4f")
        if self.config.loss_tvl_weight > 0.0:
            self.tv_loss = AverageMeter("tv_loss", ":.4f")

        self.meters = [
            self.bce_loss,
            self.ce_loss,
            self.gd_loss,
            self.tv_loss,
            self.lr,
            self.step_time,
            self.data_time,
            self.num_tokens,
            self.num_pos,
        ]

    def get_active_meters(self) -> list[AverageMeter]:
        """
        Returns a list of all meter instances that are not None. For initializing ProgressMeter
        """
        return [m for m in self.meters if m is not None]

    def reset_all(self):
        """Resets all active meters."""
        for meter in self.get_active_meters():
            meter.reset()

    def reduce_all(self):
        """Do all_reduce for all active meters (if needed)"""
        if self.should_reduce:
            for meter in self.get_active_meters():
                meter.all_reduce()

    def get_stats(self) -> dict[str, float]:
        """Returns a dict of all meter values"""
        log_dict = {}
        for meter in self.get_active_meters():
            log_dict[meter.name] = meter.val
        return log_dict

    def get_wandb_stats(self) -> dict[str, float]:
        """Returns a dict of all meter values or sums (for wandb)"""
        log_dict = self.get_stats()
        log_dict_wandb = {}
        for k, v in log_dict.items():
            item = log_dict[k]
            log_dict_wandb[f"train/{k}"] = item

        log_dict_wandb["train/bce_loss_weighted"] = self.config.loss_bce_weight * log_dict["bce_loss"]
        log_dict_wandb["train/ce_loss_weighted"] = self.config.loss_ce_weight * log_dict["ce_loss"]
        log_dict_wandb["train/gd_loss_weighted"] = self.config.loss_gd_weight * log_dict["gd_loss"]
        log_dict_wandb["train/tv_loss_weighted"] = self.config.loss_tvl_weight * log_dict["tv_loss"]

        return log_dict_wandb


def display_metrics_table(
    metrics_history: list[dict[str, float | int | None]],
    key: str = "epoch",
    title: str = " --- Metrics for Each Epoch --- ",
    monitor: str = "R1-at-0.7",
):
    """
    Generates and prints a formatted text table of metrics from a history of runs.

    Args:
        metrics_history (list[dict]): A list of dictionaries of metrics.
        key (str, optional): The row key in the metrics dictionary (e.g., "epoch"). Defaults to "epoch".
        title (str, optional): The title to print above the table.
        monitor (str, optional): The key of the primary metric to monitor. Defaults to "R1-at-0.7".
    """
    if not metrics_history:
        print("No metrics to display.")
        return

    print(title)

    all_header_keys = set()
    for metrics in metrics_history:
        all_header_keys.update(metrics.keys())

    if monitor in all_header_keys:
        all_header_keys.remove(monitor)
        list_headers = [monitor] + list(all_header_keys)
    else:
        list_headers = list(all_header_keys)

    all_header_keys.discard(key)
    sorted_headers = sorted(list_headers)

    key_width = 12
    col_width = 13

    header_str = f"{key:<{key_width}}" + "".join(f"{h:<{col_width}}" for h in sorted_headers)
    if monitor in sorted_headers:
        header_str = header_str.replace(monitor, f"*{monitor}*")
    print(header_str)
    print("-" * len(header_str))

    for metrics in metrics_history:
        key_num = metrics.get(key, "N/A")
        if isinstance(key_num, float):
            _row_str = f"{key_num:.4f}"
            row_str = f"{_row_str:<{key_width}}"
        else:
            row_str = f"{key_num:<{key_width}}"

        for header in sorted_headers:
            value = metrics.get(header, "N/A")

            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            row_str += f"{formatted_value:<{col_width}}"

        print(row_str)


def save_checkpoint_deepspeed(
    model: deepspeed.DeepSpeedEngine,
    config: Config,
    stats: dict[str, float],
    ckpt_dir: str,
    val_output: tuple[list[dict] | None, list[dict] | None] = ([], []),
):
    """
    Saves a model checkpoint using the DeepSpeed engine.

    - Saving the model and optimizer states via DeepSpeed.save_checkpoint.
    - Naming the checkpoint based on the global step and a monitored metric, if possible.
    - Saving metadata, arguments, and validation results (predictions, ground truths) as JSON files.
    - Copying the terminal log file to the checkpoint directory.
    - Logging artifacts to wandb if enabled.

    Args:
        model (deepspeed.DeepSpeedEngine): The DeepSpeed-wrapped model engine.
        config (Config): The training arguments.
        stats (dict[str, float]): A dictionary of metrics and statistics for the current step.
        ckpt_dir (str): The base directory where checkpoints are saved.
        val_output (tuple[list[dict], list[dict]]): tuple containing preds and gts of the val loop
    """
    val_preds = val_output[0]
    val_gts = val_output[1]
    all_stats = []
    global_step = model.global_steps

    if isinstance(config.monitor, str) and config.monitor in stats:
        metric_value = stats.get(config.monitor, 0.0)
        ckpt_name = f"global_step{global_step}_{config.monitor}-{metric_value:.4f}"
    else:
        ckpt_name = f"global_step{global_step}"
    try:
        model.save_checkpoint(
            ckpt_dir,
            tag=ckpt_name,
            client_state=stats,
            exclude_frozen_parameters=not config.checkpoint_save_frozen_parameters,
            save_latest=True,
        )
    except Exception as e:
        log(f"Fatal: Failed to save deepspeed checkpoint with error: \n{e}")

    if config.rank != 0:
        return

    ckpt_stats = {"args": vars(config), "stats": stats, "ds_config": model.config}

    log_src = os.path.join("runs", config.run_name, "terminal_output.log")
    log_dst_dir = os.path.join(ckpt_dir, ckpt_name)
    log_dst = os.path.join(log_dst_dir, "terminal_output.log")

    try:
        # move the logs
        if os.path.exists(log_src):
            os.makedirs(log_dst_dir, exist_ok=True)
            shutil.copy2(log_src, log_dst)

        # save the predictions and scores for offline eval
        log_json_dst = os.path.join(log_dst_dir, "stats.json")
        val_gts_dir = os.path.join(log_dst_dir, "ground_truths.json")
        val_pred_dir = os.path.join(log_dst_dir, "predictions.json")
        if os.path.isdir(log_dst_dir) and config.rank == 0:
            with open(log_json_dst, "w") as f_out:
                f_out.write(json.dumps(ckpt_stats, indent=2))
            if config.do_val:
                with open(val_gts_dir, "w") as f_out:
                    f_out.write(json_dumps(val_gts, max_inline_length=150))
                with open(val_pred_dir, "w") as f_out:
                    f_out.write(json_dumps(val_preds, max_inline_length=150))

    except Exception as e:
        log(f"Warning: Failed to copy log file to checkpoint folder: {e}")

    if config.rank == 0 and config.report_to_wandb:
        try:
            if os.path.exists(log_src):
                # wandb.save(log_src)
                artifact = wandb.Artifact(f"terminal_output_{ckpt_name}", type="log")
                artifact.add_file(log_src)
                wandb.log_artifact(artifact)
        except Exception as e:
            log(f"Warning: Failed to upload log file to wandb: {e}")

    for _dir in os.listdir(ckpt_dir):
        ckpt_global_step = -1

        # make sure it's a directory first
        full_dir = os.path.join(ckpt_dir, _dir)
        if not os.path.isdir(full_dir):
            continue

        if "global_step" in _dir:
            after_global_step = _dir.find("global_step") + len("global_step")
            _ckpt_global_step = _dir[after_global_step:].split("_")[0]
            if str.isdigit(_ckpt_global_step):
                ckpt_global_step = int(_ckpt_global_step)

        stats_json_path = os.path.join(full_dir, "stats.json")
        if not os.path.isfile(stats_json_path):
            continue

        with open(stats_json_path) as f_in:
            curr_ckpt_stats = json.loads(f_in.read())

        if "stats" in curr_ckpt_stats:
            curr_ckpt_stats = curr_ckpt_stats["stats"]
            if "global_step" not in curr_ckpt_stats:
                curr_ckpt_stats["global_step"] = ckpt_global_step
            # curr_ckpt_stats["ckpt_dir"] = _dir
            for k in list(curr_ckpt_stats.keys()):
                if "time" in k or "tokens" in k:
                    curr_ckpt_stats.pop(k)
            all_stats.append(curr_ckpt_stats)

    if config.rank == 0:
        try:
            print("\n")
            display_metrics_table(
                all_stats,
                key="global_step",
                monitor=config.monitor,
                title=" --- Checkpoint Metrics ---",
            )
            print("")
        except Exception as e:
            print(f"cannot display metrics table: {e}")


def find_and_load_checkpoint_deepspeed(
    config: Config, model: deepspeed.DeepSpeedEngine, num_micro_batch_in_epoch
) -> tuple[str, int, int, bool]:
    ok = False
    # --- Find checkpoint dir ---
    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name.replace("/", "-"))
    if not os.path.isdir(ckpt_dir):
        log(f"Checkpoint dir at {ckpt_dir} is not a valid directory")
        return ckpt_dir, 0, 0, False

    resume_from_epoch, resume_from_step = 0, 0
    if config.resume_from_latest:
        with open(f"{ckpt_dir}/latest", "r") as f:
            config.resume_from_checkpoint = f.read().strip()
            log(f"Using latest checkpoint from {ckpt_dir}/{config.resume_from_checkpoint}")

    # --- Load checkpoint ---
    if config.resume_from_checkpoint is not None:
        if os.path.isdir(os.path.join(ckpt_dir, config.resume_from_checkpoint)):
            log(
                f"Loading checkpoint from {ckpt_dir}/{config.resume_from_checkpoint} with load_module_only={config.load_module_only}"
            )
            _, client_sd = model.load_checkpoint(
                ckpt_dir,
                tag=config.resume_from_checkpoint,
                load_module_strict=False,
                load_module_only=config.load_module_only,
            )

            infer_step_from_ckpt = False
            log("Checkpoint Keys:")
            if client_sd is not None and isinstance(client_sd, dict):
                if config.rank == 0:
                    for k, v in client_sd.items():
                        if "param" in k or "buffer" in k:
                            continue
                        print(f"  {k}: {v}")

                # infer last step and last epoch from the client_sd (if any)
                if "step" in client_sd and "epoch" in client_sd:
                    resume_from_step = client_sd["step"]
                    resume_from_epoch = client_sd["epoch"]
                    infer_step_from_ckpt = True
                elif "epoch" in client_sd:  # if only epoch information exists, assume it's the last batch of the epoch
                    resume_from_step = num_micro_batch_in_epoch - 1
                    resume_from_epoch = client_sd["epoch"]
                    infer_step_from_ckpt = True

            if not infer_step_from_ckpt:
                resume_from_step = (
                    model.global_steps * config.gradient_accumulation_steps - 1
                )  # if nothing is given, assume it is the last step of the global step
                resume_from_epoch = resume_from_step // num_micro_batch_in_epoch

            # increase the epoch if step is at the end of epoch
            if (resume_from_step + 1) % num_micro_batch_in_epoch == 0:
                resume_from_epoch += 1  # end of epoch, continue to next epoch

            # step === step modulo (step in epoch)
            resume_from_step -= num_micro_batch_in_epoch * resume_from_epoch
            resume_from_step += 1  # increase step

            ok = True

    return ckpt_dir, resume_from_step, resume_from_epoch, ok


def load_pretrained_state_dict(model: torch.nn.Module, path: str, bad_key: str = ""):
    tensors = {}
    if path.endswith("safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                good_key = key.replace(bad_key, "")
                tensors[good_key] = f.get_tensor(key)
    else:
        _tensors = torch.load(path)
        for k, v in _tensors.items():
            good_k = k.replace(bad_key, "")
            tensors[good_k] = v

    results = model.load_state_dict(tensors)
    rank = os.environ.get("RANK", 0)

    print(f"Rank {rank} missing keys:", results.missing_keys)
