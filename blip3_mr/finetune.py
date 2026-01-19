"""Main finetuning script for training with deepspeed. Can be also used for validation only (with or without deepspeed)"""

from typing import Any, cast

import deepspeed
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from blip3_mr.config import (
    get_config,
)
from blip3_mr.dataset import make_test_datainfo, make_train_val_datainfos
from blip3_mr.model import create_model, wrap_model_in_deepspeed
from blip3_mr.test import test_one_epoch
from blip3_mr.train import deepspeed_finetune_one_epoch_generator
from blip3_mr.utils import (
    ProgressMeter,
    TrainingMeters,
    calculate_loss_weight,
    init_wandb,
    log,
    random_seed,
    save_checkpoint_deepspeed,
    unwrap_model,
)
from blip3_mr.validate import (
    calc_val_steps,
    validate_one_epoch,
)

# Reset when dataloader step + 1 % max_meter_step == 0 to prevent float overflow
MAX_METER_STEP = 256
COMPILE_MODE = "default"


def main():
    # --- Setup
    config = get_config()
    device = torch.device(f"cuda:{config.rank}")
    random_seed(config.seed)
    if config.world_size > 1 and config.deepspeed:
        deepspeed.init_distributed()
    if config.report_to_wandb and config.do_train:
        init_wandb(config)
    # ---

    # --- Base model
    log(f"Loading base model and tokenizer. Using local model: {config.use_local_model}")
    model, tokenizer = create_model(config)

    # --- Setup dataloaders
    log("Loading and testing datasets and dataloaders")
    train_datainfo, val_datainfo = make_train_val_datainfos(
        tokenizer=tokenizer, config=config, distributed=config.world_size > 1, verbose=False
    )
    test_datainfo = make_test_datainfo(tokenizer, config, verbose=False) if config.do_test else None

    num_micro_batch_in_epoch = len(train_datainfo.dataloader)
    num_global_steps = (num_micro_batch_in_epoch * config.num_epochs) // config.gradient_accumulation_steps
    log(f"Micro batch per epoch: {num_micro_batch_in_epoch}")

    # --- Val and/or test single GPU, vanilla pytorch---
    if not (config.deepspeed or config.do_train):
        model = model.to(device)
        model.requires_grad_(False)

        if config.do_val:
            validate_one_epoch(config, model, val_datainfo, do_save=True, output_dir=f"runs/{config.run_name}")

        if config.do_test:
            test_one_epoch(
                model,
                test_datainfo.dataloader,  # type: ignore
                config.training_precision,
                generation_kwargs={"num_beams": config.num_val_beams},
                tokenizer=tokenizer,
                output_dir=f"runs/{config.run_name}",
                num_frames=test_datainfo.num_frames,
            )
        return

    # --- Initialize deepspeed ---
    orig_mod = unwrap_model(model)
    ckpt_dir, resume_from_step, resume_from_epoch, deepspeed_model = wrap_model_in_deepspeed(
        config, model, num_micro_batch_in_epoch, num_global_steps
    )
    model = deepspeed_model

    if not config.vision_tokenizer_train:
        model.vision_tokenizer.requires_grad_(False)
    if not config.lang_model_train:
        model.lang_model.requires_grad_(False)

    if config.gradient_checkpointing:
        log("Initializing gradient checkpointing")
        orig_mod.init_gradient_checkpointing()

    log("Compiling model")
    orig_mod.vision_encoder.compile(mode=COMPILE_MODE)
    orig_mod.vision_tokenizer.compile(mode=COMPILE_MODE)
    orig_mod.lang_model.compile(mode=COMPILE_MODE)

    if config.rank == 0:
        print("Trainable parameters:")
        print(orig_mod.num_trainable_params_per_module)

    # --- Val loop with deepspeed
    if not config.do_train:
        validate_one_epoch(config, model, val_datainfo, do_save=True, output_dir=f"runs/{config.run_name}")
        deepspeed.dist.destroy_process_group()
        return

    # --- Normal Training starts here
    if config.rank == 0:
        log("Steps: ")
        print(f"Total global steps : {num_global_steps}")
        print(f"Resume from step   : {resume_from_step}")
        print(f"Resume from epoch  : {resume_from_epoch}")

    # --- Sanity checking ---
    if config.float_sanity_epoch > 0:
        num_sanity_steps = config.float_sanity_epoch * len(val_datainfo.dataloader)
        config.num_sanity_steps = int((num_sanity_steps // config.world_size) * config.world_size)
        deepspeed.dist.barrier()
        log(f"Performing sanity check for {config.num_sanity_steps} steps")
        validate_one_epoch(config, model, val_datainfo, max_iter=config.num_sanity_steps)
        torch.cuda.empty_cache()

    # --- Calculate at what steps to do eval
    val_steps = calc_val_steps(config.num_epochs, num_micro_batch_in_epoch, config.num_val_per_epoch)
    log(f"Validation steps:\n{', '.join([str(v) for v in val_steps])}")
    val_step = -1

    # --- Training loop ---
    for epoch in range(resume_from_epoch, config.num_epochs):
        current_loss_weight = calculate_loss_weight(config, epoch)
        if config.rank == 0 and config.report_to_wandb:
            wandb.log(current_loss_weight, step=model.global_steps)

        training_meters = TrainingMeters(config)
        progress_meter = None
        if config.rank == 0:
            progress_meter = ProgressMeter(
                num_batches=num_micro_batch_in_epoch * config.num_epochs - 1,
                meters=training_meters.get_active_meters(),
            )

        latest_val_preds, latest_val_truths = [], []

        # Build generator
        finetune_step_generator = deepspeed_finetune_one_epoch_generator(
            config=config,
            resume_from_step=resume_from_step,
            model=model,
            epoch=epoch,
            datainfo=train_datainfo,
            meters=training_meters,
            device=torch.device(f"cuda:{config.rank}"),
        )
        global_step_iterator = tqdm(
            finetune_step_generator,
            disable=config.rank != 0,
            total=num_micro_batch_in_epoch,
            initial=resume_from_step,
            desc=f"Run training on epoch: {epoch}",
            ncols=100,
        )

        all_stats = {}

        for global_step, step, local_step in global_step_iterator:
            should_log = (step + 1) % config.logging_steps == 0
            if should_log:
                # log and if accumulated, do log to wandb. logging step should have a common factor with accum_step
                training_meters.reduce_all()
                if config.rank == 0:
                    progress_meter.display(step)

                    should_wandb_log = config.report_to_wandb and ((step + 1) % config.gradient_accumulation_steps == 0)
                    if should_wandb_log:
                        train_stats = training_meters.get_stats()
                        all_stats = all_stats | train_stats
                        train_wandb_stats = {
                            "train/global_step": global_step,
                            "train/epoch": epoch,
                            "train/step": step,
                            **training_meters.get_wandb_stats(),
                        }
                        wandb.log(train_wandb_stats, step=global_step)

            if (local_step + 1) % MAX_METER_STEP == 0:
                training_meters.reset_all()

            # display summaries in the last step of epoch
            is_last_epoch_step = (step + 1) % num_micro_batch_in_epoch == 0
            if is_last_epoch_step:
                training_meters.reduce_all()
                if config.rank == 0:
                    progress_meter.display_summary()

            # do val, can be one time or multiple times throughout the epoch
            do_val = config.do_val and (step in val_steps)
            if do_val:
                torch.distributed.barrier()
                torch.cuda.empty_cache()

                log(f"Start validation loop on epoch: {epoch} and global step: {global_step}")
                val_outputs = validate_one_epoch(config=config, model=model, dataset=val_datainfo)
                val_stats = val_outputs["metrics"]
                val_preds = val_outputs["predictions"]
                val_truths = val_outputs["ground_truths"]
                val_wandb_stats = {f"val/{k}": v for k, v in val_stats}

                # gather all preds on rank 0
                if config.world_size > 1:
                    all_preds = [None for _ in range(config.world_size)]
                    all_gts = [None for _ in range(config.world_size)]

                    dist.all_gather_object(all_preds, val_preds)
                    dist.all_gather_object(all_gts, val_truths)

                    gather_ok = all([i is not None for i in all_preds]) and all([i is not None for i in all_gts])

                    if config.rank == 0 and gather_ok:
                        all_preds = cast(list[list[Any]], all_preds)
                        all_gts = cast(list[list[Any]], all_gts)
                        latest_val_preds = [item for sublist in all_preds for item in sublist]
                        latest_val_truths = [item for sublist in all_gts for item in sublist]

                all_stats = all_stats | val_stats
                if config.report_to_wandb and config.rank == 0:
                    wandb.log(val_wandb_stats, step=global_step)
                val_step += 1
                training_meters.reset_all()

            do_save_after_val = do_val and (val_step + 1) % config.checkpoint_every_n_val == 0
            do_save_end_epoch = ((epoch + 1) % config.checkpoint_every_n_epoch == 0) and is_last_epoch_step

            do_save = do_save_after_val or do_save_end_epoch
            if do_save:
                all_stats["step"] = step
                all_stats["epoch"] = epoch
                val_outputs = (latest_val_preds, latest_val_truths)
                save_checkpoint_deepspeed(model, config, all_stats, ckpt_dir, val_outputs)

        # after continuing, the next epochs should all start from 0
        resume_from_step = 0
        torch.distributed.barrier()
    # ---

    if config.report_to_wandb and config.rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
