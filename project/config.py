import argparse
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class Config:
    # Base model
    base_model_name_or_path: str | None = None
    use_local_model: bool = False
    gradient_checkpointing: bool = True

    # Vision Tokenizer
    vision_tokenizer_train: bool = False
    vision_tokenizer_lora: bool = False
    vision_tokenizer_pretrained: str | None = None

    # Lang Model
    lang_model_train: bool = False
    lang_model_lora: bool = False
    lang_model_pretrained: str | None = None
    lang_model_adapter: str | None = None

    # pooler
    pooler_type: str = "Conv1D"
    pooler_stride: int = 4

    # Training args
    run_name: str = "main"
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    training_precision: Literal["amp_bf16", "bf16", "fp32", "fp16", "amp_fp16"] = "bf16"
    num_epochs: int = 20
    float_sanity_epoch: float = 0.0
    num_sanity_steps: int = 0
    num_val_beams: int = 1
    weight_decay: float = 0.0
    do_train: bool = False
    do_val: bool = False
    do_test: bool = False
    num_val_per_epoch: int = 1

    # lora
    lora: bool = False
    lora_r: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = True
    init_lora_weights: str = "gaussian"

    # Losses
    loss_mapping_path: str = ""
    # These four should be loaded from the loss_mapping_path file, not from argparse
    loss_ce_weight: float = 0.0
    loss_gd_weight: float = 0.0
    loss_bce_weight: float = 0.0
    loss_tv_weight: float = 0.0
    #
    bce_pos_weight: float = 1.0
    soft_loss: bool = True
    tv_beta: float = 0.7

    # Checkpointing
    checkpoint_dir: str = "runs"
    checkpoint_every_n_epoch: int = 1
    resume_from_checkpoint: Optional[str] = None
    resume_from_latest: bool = False
    monitor: str = "R1-at-0.7"  #
    load_module_only: bool = False  # Load only the model weights
    checkpoint_save_frozen_parameters: bool = False
    checkpoint_every_n_val: int = 1

    # Data args
    base_data_dir: str = "datasets"
    dataset_name: str = "qvhighlights"
    test_dataset_name: str = "qvhighlights-test"
    dataset_config: str = "config.yaml"
    train_micro_batch_size_per_gpu: int = 8
    val_batch_size: int = 0
    num_train_workers: int = 4
    num_val_workers: int = 4
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0

    # Distributed training args
    # no_set_device_rank: bool = False
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1

    # Logging
    logging_steps: int = 8
    report_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Others
    seed: int = 53

    # Deepspeed
    deepspeed: bool = False
    deepspeed_config: str | None = None
    deepspeed_from_universal: bool = False


@dataclass
class DataConfig:
    # Data args
    base_data_dir: str = "datasets"
    dataset_name: str = "qvhighlights"
    test_dataset_name: str = "qvhighlights-test"
    yaml_path: str = "config.yaml"
    train_micro_batch_size_per_gpu: int = 8
    val_batch_size: int = 0
    num_train_workers: int = 4
    num_val_workers: int = 4
    num_train_samples: int = 0
    num_val_samples: int = 0
    num_test_samples: int = 0


# fmt: off
def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BLIP-3 Training Arguments")
    group = parser.add_argument_group("Base Model")
    group.add_argument("--base_model_name_or_path", type=str, default=None)
    group.add_argument('--use_local_model', action="store_true", default=False)
    group.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save memory.")

    group = parser.add_argument_group("Vision Tokenizer")
    group.add_argument("--vision_tokenizer_train", action="store_true", default=False, help="Flag to train the vision tokenizer.")
    group.add_argument("--vision_tokenizer_lora", action="store_true", default=False, help="Flag to wrap the vision tokenizer in LoRA.")
    group.add_argument("--vision_tokenizer_pretrained", type=str, default=None)

    group = parser.add_argument_group("Language Model")
    group.add_argument("--lang_model_train", action="store_true", default=False, help="Flag to train the language model.")
    group.add_argument("--lang_model_lora", action="store_true", default=False, help="Flag to wrap the language model in LoRA.")
    group.add_argument("--lang_model_pretrained", type=str, default=None)
    group.add_argument("--lang_model_adapter", type=str, default=None)

    group = parser.add_argument_group("Training")
    group.add_argument("--run_name", type=str, default="main", help="Name for the training run, used for logging and saving.")
    group.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    group.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    group.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for the LR scheduler.")
    group.add_argument("--training_precision", type=str, default="amp_bf16", choices=["amp_bf16", "bf16", "fp32", "fp16", "amp_fp16"], help="Training precision.")
    group.add_argument("--num_epochs", type=int, default=4, help="Total number of training epochs.")
    group.add_argument("--float_sanity_epoch", type=float, default=0.0, help="Run a sanity check on a fraction of the validation set before training. Use this or num_sanity_steps")
    group.add_argument("--num_sanity_steps", type=int, default=0, help="Number of sanity check steps to run before training. Use this or float_sanity_epoch")
    group.add_argument("--num_val_beams", type=int, default=1, help="Number of beams for beam search during validation.")
    group.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for all trainable parameters.")
    group.add_argument("--do_val", default=False, action="store_true")
    group.add_argument("--do_train", default=False, action="store_true")
    group.add_argument("--do_test", default=False, action="store_true")
    group.add_argument("--num_val_per_epoch", type=int, default=1)

    group = parser.add_argument_group("Peft")
    group.add_argument("--lora_r", type=int, default=16, choices=[8, 16, 32], help="The rank of the LoRA matrices.")
    group.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers.")
    group.add_argument("--use_rslora", action="store_true", default=False)
    group.add_argument("--init_lora_weights", type=str, default="gaussian")

    group = parser.add_argument_group("Loss")
    group.add_argument("--loss_mapping_path", type=str, default="")
    group.add_argument("--bce_pos_weight", type=float, default=1.0, help="Positive weight for the binary cross-entropy loss.")
    group.add_argument("--soft_loss", default=False, action="store_true", help="Use soft prediction for loss")
    group.add_argument("--gd_norm", type=str, choices=["square", "linear"], default="square")
    group.add_argument("--tvl_beta", type=float, default=0.5, help="Beta parameter for Tversky Loss")

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--checkpoint_dir", type=str, default='runs', help="Directory to save model checkpoints. Defaults to 'runs'")
    group.add_argument("--checkpoint_every_n_epoch", type=int, default=1, help="Save a checkpoint every N epochs (0 to disable).")
    group.add_argument("--resume_from_checkpoint", type=str, help="Path to a checkpoint to resume training from.", default=None)
    group.add_argument("--monitor", type=str, default="R1-at-0.7", help="Metric to monitor for best checkpoint selection (e.g., 'val_loss', 'accuracy'). If None, saves latest checkpoint and prunes earliest.")
    group.add_argument("--load_module_only", action="store_true", default=False, help="If set, only save model weights in checkpoints (no optimizer/lr_scheduler state).")
    group.add_argument("--checkpoint_save_frozen_parameters", action="store_true", default=False, help="If set, save frozen parameters in checkpoints.")
    group.add_argument("--resume_from_latest", action="store_true", default=False, help="If set, resume from the latest checkpoint.")
    group.add_argument("--checkpoint_every_n_val", type=int, default=0)
    group.add_argument("--deepspeed_from_universal", action="store_true", default=False)

    group = parser.add_argument_group("Data")
    group.add_argument("--base_data_dir", type=str, default="datasets", help="Base directory where datasets are stored.")
    group.add_argument("--dataset_name", type=str, default="qvhighlights", help="Name of the dataset to use.")
    group.add_argument("--test_dataset_name", type=str, default="qvhighlights-test", help="Name of the test dataset to use. If none, then will use --dataset_name")
    group.add_argument("--dataset_config", type=str, default="config.yaml", help="Configuration file of datasets")
    group.add_argument("--train_micro_batch_size_per_gpu", type=int, default=8, help="Training batch size per GPU")
    group.add_argument("--val_batch_size", type=int, default=0, help="Validation batch size (defaults to training batch size if 0).")
    group.add_argument("--num_train_workers", type=int, default=4, help="Number of workers for the Train DataLoader.")
    group.add_argument("--num_val_workers", type=int, default=4, help="Number of workers for the Val DataLoader.")
    group.add_argument("--num_train_samples", type=int, default=0, help="Number of training samples to use per epoch. Set to 0 to use all")
    group.add_argument("--num_val_samples", type=int, default=0, help="Number of validation samples to use. Set to 0 to use all")
    group.add_argument("--num_test_samples", type=int, default=0, help="Number of test samples to use. Set to 0 to use all")
    group.add_argument("--sampler", type=str, default='pytorch', help="Choose which sampler to use")

    group = parser.add_argument_group("Distributed Training")
    group.add_argument("--local_rank", default=0, type=int, help="Local rank for distributed training.")

    group = parser.add_argument_group("Logging")
    group.add_argument("--logging_steps", type=int, default=8, help="Log training loss every N steps.")
    group.add_argument("--report_to_wandb", action="store_true", default=False, help="Report metrics to Weights & Biases.")
    group.add_argument("--wandb_project", type=str, default=None, help="W&B project name.")
    group.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team or username).")

    group = parser.add_argument_group("Other")
    group.add_argument("--seed", type=int, default=53, help="Random seed for reproducibility.")

    group = parser.add_argument_group("DeepSpeed")
    group.add_argument("--deepspeed", default=False, action="store_true", help="Flag whether deepspeed is used or not")
    group.add_argument("--deepspeed_config", type=str, default=None, help="Path to deepspeed config file (JSON)")
    return parser
# fmt: on


def print_config(parser: argparse.ArgumentParser, config: Config):
    """
    Prints parsed arguments by group and any additional attributes.
    """
    if config.rank != 0:
        return
    print("--- Parsed Arguments by Group ---")
    parsed_arg_names = set()

    for group in parser._action_groups:
        print(f"\n--- {group.title} ---")
        group_args_printed = False
        for action in group._group_actions:
            if hasattr(config, action.dest) and action.dest is not None:
                print(f"  {action.dest}: {getattr(config, action.dest)}")
                parsed_arg_names.add(action.dest)
                group_args_printed = True

        if not group_args_printed:
            if group.title not in ["positional arguments", "options"] or len(group._group_actions) > 0:
                print("  (No arguments explicitly defined in this group)")

    print("\n--- Other Attributes Added After Parsing ---")
    other_attributes_printed = False
    for key, value in vars(config).items():
        if key not in parsed_arg_names:
            print(f"  {key}: {value}")
            other_attributes_printed = True

    if not other_attributes_printed:
        print("  (No additional attributes found)")


def get_deepspeed_config_from_config(config: Config, total_training_global_steps: int) -> dict[str, int]:
    assert config.deepspeed_config is not None, "deepspeed_config is required"
    with open(config.deepspeed_config, "r") as f_in:
        ds_conf = json.load(f_in)

    ds_conf["train_micro_batch_size_per_gpu"] = config.train_micro_batch_size_per_gpu
    ds_conf["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    if "scheduler" in ds_conf:
        ds_conf["scheduler"]["params"]["total_num_steps"] = total_training_global_steps
        ds_conf["scheduler"]["params"]["warmup_num_steps"] = (
            config.warmup_steps if config.warmup_steps > 0 else total_training_global_steps // 10
        )
    if config.training_precision == "bf16":
        ds_conf["bf16"]["enabled"] = True
        del ds_conf["fp16"]
    elif config.training_precision == "fp16":
        ds_conf["fp16"]["enabled"] = True
        del ds_conf["bf16"]
    if "optimizer" in ds_conf:
        if config.weight_decay > 0.0:
            ds_conf["optimizer"]["params"]["weight_decay"] = config.weight_decay
        ds_conf["optimizer"]["params"]["lr"] = config.learning_rate

    if config.deepspeed_from_universal:
        if "checkpoint" not in ds_conf:
            ds_conf["checkpoint"] = {"load_universal": True}
        elif isinstance(ds_conf["checkpoint"], dict):
            ds_conf["checkpoint"]["load_universal"] = True

    return ds_conf


def config_check(config: Config):
    """
    Perform checks and corrections on the given args (in-place).
    Can be done in every rank.
    """
    if torch.cuda.get_device_capability(0)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    config.rank = int(os.environ.get("RANK", 0))
    config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))

    config.lora = any([config.vision_tokenizer_lora, config.lang_model_lora])

    if config.checkpoint_dir is None:
        config.checkpoint_dir = "runs"

    loss_weights = [
        config.loss_ce_weight,
        config.loss_bce_weight,
        config.loss_gd_weight,
        config.loss_tv_weight,
    ]
    total_loss_weight = sum(loss_weights)

    if total_loss_weight < 0.99999 or total_loss_weight > 1.00001:
        if total_loss_weight == 0.0:
            loss_weights = [1 / len(loss_weights) for _ in loss_weights]
        else:
            loss_weights = [loss / total_loss_weight if total_loss_weight > 0.0 else 0.0 for loss in loss_weights]
        config.loss_ce_weight = loss_weights[0]
        config.loss_bce_weight = loss_weights[1]
        config.loss_gd_weight = loss_weights[2]
        config.loss_tv_weight = loss_weights[3]

    if config.do_val:
        assert config.val_batch_size != 0, "val batch size can't be 0"

    if config.num_sanity_steps > 0 and config.float_sanity_epoch > 0.0:
        config.num_sanity_steps = 0

    if config.lang_model_lora:
        config.lang_model_train = True

    if config.vision_tokenizer_lora:
        config.vision_tokenizer_train = True

    if not config.do_train:
        config.report_to_wandb = False


def get_config() -> Config:
    parser = get_argument_parser()
    config = Config()
    parser.parse_args(namespace=config)
    config_check(config)
    print_config(parser, config)
    return config
