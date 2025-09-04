"""Main finetuning script for training with deepspeed. Can be also used for validation only (with or without deepspeed)"""

from typing import cast

import deepspeed
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.config import (
    get_config,
    get_deepspeed_config_from_config,
)
from blip3_mr.dataset import make_test_datainfo, make_train_val_datainfos
from blip3_mr.lora import (
    load_adapter,
)
from blip3_mr.open_flamingo.src.factory import create_model_and_tokenizer
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.test import test_one_epoch
from blip3_mr.train import deepspeed_finetune_one_epoch_generator
from blip3_mr.utils import (
    ProgressMeter,
    TrainingMeters,
    calculate_loss_weight,
    find_and_load_checkpoint_deepspeed,
    init_wandb,
    isfile,
    load_pretrained_state_dict,
    log,
    random_seed,
    save_checkpoint_deepspeed,
    unwrap_model,
)
from blip3_mr.validate import (
    calc_val_steps,
    save_val_result_to_dirs,
    validate_one_epoch_v2,
)

# Reset when dataloader step + 1 % max_meter_step == 0 to prevent float overflow
MAX_METER_STEP = 128

# Torch compile mode for the submodels
COMPILE_MODE = "default"

"""
# Save the base model weights to use the local model
import os
import torch
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5", trust_remote_code=True
).vlm
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/xgenmm.pt")
"""


def main():
    # --- Setup
    config = get_config()
    device = torch.device(f"cuda:{config.rank}")
    random_seed(config.seed)
    if config.world_size > 1 and config.deepspeed:
        deepspeed.init_distributed()
    if config.report_to_wandb and config.do_train:
        log("Initialize wandb logging")
        init_wandb(config)
    # ---

    # --- Base model
    log(f"Loading base model and tokenizer. Using local model: {config.use_local_model}")
    # the hf model can't be used for forward(), monkey patching also not working
    if config.use_local_model or config.do_train:
        model, tokenizer = create_model_and_tokenizer(
            gradient_checkpointing=config.gradient_checkpointing, pretrained=config.base_model_name_or_path
        )
    else:
        hf_model = AutoModelForVision2Seq.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
            trust_remote_code=True,
            use_fast=False,
            legacy=False,
        )
        tokenizer = hf_model.update_special_tokens(tokenizer)
        model = hf_model.vlm
        if config.rank == 0:
            print(hf_model.config)
        model = cast(XGenMMPerceiver, model)
        if config.do_train:
            model.set_trainable()
            if config.rank == 0:
                print("Trainable parameters:")
                print(model.num_trainable_params_per_module)
    # ---

    # --- Setup dataloaders
    log("Loading and testing datasets and dataloaders")
    train_datainfo, val_datainfo = make_train_val_datainfos(
        tokenizer=tokenizer, config=config, distributed=config.world_size > 1, verbose=True
    )
    if config.do_test:
        test_datainfo = make_test_datainfo(tokenizer, config)
    else:
        test_datainfo = None
    num_micro_batch_in_epoch = len(train_datainfo.dataloader)
    num_global_steps = (num_micro_batch_in_epoch * config.num_epochs) // config.gradient_accumulation_steps
    # ---

    # --- Load pretrained with state_dict instead of from_pretrained since there are modifications
    # to the base models' submodel both in __init__ of XGenMMPerceiver and .from_pretrained
    if not config.vision_tokenizer_lora and config.vision_tokenizer_pretrained is not None:
        log(f"Loading vision_tokenizer from {config.vision_tokenizer_pretrained}")
        load_pretrained_state_dict(
            model.vision_tokenizer,
            config.vision_tokenizer_pretrained,  # type: ignore
            bad_key="vision_tokenizer.",
        )

    if not config.lang_model_lora and isfile(config.lang_model_pretrained):
        load_pretrained_state_dict(model.lang_model, config.lang_model_pretrained, bad_key="lang_model.")  # type: ignore
        log(f"Loading lang_model from {config.lang_model_pretrained}")
    # ---

    # --- Initialize gradient checkpointing here
    if config.gradient_checkpointing:
        log("Initializing gradient checkpointing")
        model.init_gradient_checkpointing()
    # ---

    # --- Load adapters
    if config.lang_model_lora:
        model.lang_model = load_adapter(
            model.lang_model,
            config,
            task_type="CAUSAL_LM",
            target_modules="phi3",
            model_name_or_path=config.lang_model_pretrained,
        )
        model.lang_model.to(torch.bfloat16)
        log(f"Loaded lang_model adapter from {config.lang_model_pretrained} with config:")
        if config.rank == 0:
            print(model.lang_model.peft_config)
        model.lang_model.print_trainable_parameters()

    if config.vision_tokenizer_lora:
        model.vision_tokenizer = load_adapter(
            # @TODO fix the vision tokenizer since it's a plain torch.nn.Module
            model.vision_tokenizer,
            config,
            task_type=None,
            target_modules="all-linear",
            model_name_or_path=config.vision_tokenizer_pretrained,
        )
        model.vision_tokenizer.bfloat16()
        log(f"Loaded vision_tokenizer adapter from {config.vision_tokenizer_pretrained} with config:")
        if config.rank == 0:
            for k, v in model.vision_tokenizer.peft_config["default"].items():
                print(f"\t{k}: {v}")
        model.vision_tokenizer.print_trainable_parameters()
    # ---

    if config.rank == 0:
        print("Trainable parameters:")
        print(model.num_trainable_params_per_module)

    # --- Compile ---
    model.vision_encoder.compile(mode=COMPILE_MODE)
    model.vision_tokenizer.compile(mode=COMPILE_MODE)
    model.lang_model.compile(mode=COMPILE_MODE)
    # ---

    # --- Val and/or test without deepspeed---
    if not (config.deepspeed or config.do_train):
        model = model.to(device)
        
        log("Freezing the model")
        model.requires_grad_(False)
        if config.rank == 0:
            print("Trainable parameters:")
            print(model.num_trainable_params_per_module)

        if config.do_val:
            val_results = validate_one_epoch_v2(config=config, model=model, dataset=val_datainfo)
            if config.rank == 0:
                save_val_result_to_dirs([f"runs/{config.run_name}"], val_results)

        if config.do_test:
            test_one_epoch(
                model,
                test_datainfo.dataloader,
                config.training_precision,
                generation_kwargs={"num_beams": config.num_val_beams},
                tokenizer=tokenizer,
                output_dir=f"runs/{config.run_name}",
            )
        return
    # ---

    # --- Initialize deepspeed ---
    orig_mod = unwrap_model(model)
    deepspeed_config = get_deepspeed_config_from_config(config, num_global_steps)
    model, _, _, _ = deepspeed.initialize(model=model, config=deepspeed_config)
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    ckpt_dir, resume_from_step, resume_from_epoch, ckpt_ok = find_and_load_checkpoint_deepspeed(
        config, model, num_micro_batch_in_epoch
    )

    if ckpt_ok and config.lora and config.deepspeed_from_universal:
        if config.lang_model_lora:
            log("Reloading lora model for universal checkp0oint")
            model.module.lang_model = load_adapter(model.module.lang_model, config)

    if config.rank == 0:
        print("Trainable parameters:")
        print(orig_mod.num_trainable_params_per_module)

    # --- Val loop with deepspeeed (or deepspeed checkpoints)
    if not config.do_train:
        val_results = validate_one_epoch_v2(
            config=config, model=model, dataset=val_datainfo
        )
        if config.rank == 0:
            save_val_result_to_dirs([f"runs/{config.run_name}"], val_results)
        return
    # ---

    if config.rank == 0:
        log(f"Steps: ")
        print(f"Total global steps: {num_global_steps}")
        print(f"Resume from step: {resume_from_step}")
        print(f"Resume from epoch: {resume_from_epoch}")

    # --- Sanity checking ---
    if config.float_sanity_epoch > 0:
        num_sanity_steps = config.float_sanity_epoch * len(val_datainfo.dataloader)
        config.num_sanity_steps = int((num_sanity_steps // config.world_size) * config.world_size)
        deepspeed.dist.barrier()
        log(f"Performing sanity check for {config.num_sanity_steps} steps")
        validate_one_epoch_v2(
            config=config,
            model=model,
            dataset=val_datainfo,
            max_steps=config.num_sanity_steps,
        )
        torch.cuda.empty_cache()
    # ---

    # --- Calculate at what steps to do eval
    val_steps = calc_val_steps(config.num_epochs, num_micro_batch_in_epoch, config.num_val_per_epoch)
    if config.rank == 0:
        log("Validation steps:")
        val_steps_print = [str(v) for v in val_steps]
        print(", ".join(val_steps_print))

    val_step = -1
    # ---

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
            device=model.device,
        )
        global_step_iterator = tqdm(
            finetune_step_generator,
            disable=config.rank != 0,
            total=num_micro_batch_in_epoch,
            initial=resume_from_step,
            desc=f"Run training on epoch: {epoch}",
            ncols=120,
        )

        all_stats = {}

        for global_step, step, local_step in global_step_iterator:
            should_log = (step + 1) % config.logging_steps == 0
            if should_log:
                # log and if accumulated, do log to wandb. logging step should have a common factor with accum_step
                training_meters.reduce_all()
                if config.rank == 0:
                    progress_meter.display(step)

                    should_wandb_log = config.report_to_wandb and (
                        (step + 1) % config.gradient_accumulation_steps == 0
                    )
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
                val_outputs = validate_one_epoch_v2(
                    config=config, model=model, dataset=val_datainfo
                )
                val_stats = val_outputs["metrics"]
                val_preds = val_outputs['predictions']
                val_truths = val_outputs['ground_truths']
                val_wandb_stats = {f"val/{k}": v for k, v in val_stats}

                # gather all preds on rank 0
                if config.world_size > 1:
                    all_preds = [None for _ in range(config.world_size)]
                    all_gts = [None for _ in range(config.world_size)]

                    dist.all_gather_object(all_preds, val_preds)
                    dist.all_gather_object(all_gts, val_truths)

                    gather_ok = all([i is not None for i in all_preds]) and all(
                        [i is not None for i in all_gts]
                    )

                    if config.rank == 0 and gather_ok:
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
