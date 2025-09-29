from typing import cast

import deepspeed
import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.config import Config, get_deepspeed_config_from_config
from blip3_mr.lora import (
    load_adapter,
)
from blip3_mr.open_flamingo.src.factory import create_model_and_tokenizer
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.utils import (
    find_and_load_checkpoint_deepspeed,
    isfile,
    load_pretrained_state_dict,
    log,
    isdir,
    is_file_or_dir
)

COMPILE_MODE = "default"


def load_model(config: Config) -> tuple[XGenMMPerceiver, PreTrainedTokenizer]:
    if config.use_local_model:
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

    if is_file_or_dir(config.vision_tokenizer_pretrained):
        log(f"Loading vision_tokenizer from {config.vision_tokenizer_pretrained}")
        load_pretrained_state_dict(
            model.vision_tokenizer,
            config.vision_tokenizer_pretrained,  # type: ignore
            bad_key="vision_tokenizer.",
        )

    if isfile(config.lang_model_pretrained) or isdir(config.lang_model_pretrained):
        log(f"Loading lang_model from {config.lang_model_pretrained}")
        load_pretrained_state_dict(model.lang_model, config.lang_model_pretrained, bad_key="lang_model.")  # type: ignore

    return model, tokenizer


def wrap_model_in_lora(config: Config, model: XGenMMPerceiver, tokenizer: PreTrainedTokenizer):
    if config.lang_model_lora:
        if is_file_or_dir(config.lang_model_pretrained) and isdir(config.lang_model_adapter):
            peft_model = load_adapter(
                model.lang_model,
                config,
                model_name_or_path=config.lang_model_adapter
            )
            peft_model.to(torch.bfloat16)
            log(f"Loaded lang_model adapter from {config.lang_model_adapter} with config:")
        else:

            peft_model = load_adapter(
                model.lang_model,
                config,
                task_type="CAUSAL_LM",
                target_modules="phi3",
            )
            peft_model.to(torch.bfloat16)
            log(f"Created lang_model adapter with config:")
        
        model.lang_model = peft_model
        if config.rank == 0:
            print(peft_model.peft_config)
        peft_model.print_trainable_parameters()

    if config.vision_tokenizer_lora:
        peft_model2 = load_adapter(
            # @TODO fix the vision tokenizer since it's a plain torch.nn.Module
            model.vision_tokenizer,
            config,
            task_type=None,
            target_modules="all-linear",
            model_name_or_path=config.vision_tokenizer_pretrained,
        )
        peft_model2.bfloat16()
        model.vision_tokenizer = peft_model2
        log(f"Loaded vision_tokenizer adapter from {config.vision_tokenizer_pretrained} with config:")
        if config.rank == 0:
            for k, v in peft_model2.peft_config["default"].items():
                print(f"\t{k}: {v}")
        peft_model2.print_trainable_parameters()

    if config.rank == 0:
        print("Trainable parameters:")
        print(model.num_trainable_params_per_module)


def wrap_model_in_deepspeed(
    config: Config,
    model: XGenMMPerceiver,
    num_micro_batch_in_epoch: int,
    num_global_steps: int,
):
    deepspeed_config = get_deepspeed_config_from_config(config, num_global_steps)
    deepspeed_model, _, _, _ = deepspeed.initialize(model=model, config=deepspeed_config)
    assert isinstance(deepspeed_model, deepspeed.DeepSpeedEngine)

    ckpt_dir, resume_from_step, resume_from_epoch, ckpt_ok = find_and_load_checkpoint_deepspeed(
        config, deepspeed_model, num_micro_batch_in_epoch
    )

    if ckpt_ok and config.lora and config.deepspeed_from_universal:
        if config.lang_model_lora:
            log("Reloading lora model for universal checkp0oint")
            model.lang_model = load_adapter(model.lang_model, config)
    return ckpt_dir, resume_from_step, resume_from_epoch, deepspeed_model


def create_model(config: Config) -> tuple[XGenMMPerceiver, PreTrainedTokenizer]:
    model, tokenizer = load_model(config)
    wrap_model_in_lora(config, model, tokenizer)

    # if config.gradient_checkpointing:
    #     log("Initializing gradient checkpointing")
    #     model.init_gradient_checkpointing()

    # model.vision_encoder.compile(mode=COMPILE_MODE)
    # model.vision_tokenizer.compile(mode=COMPILE_MODE)
    # model.lang_model.compile(mode=COMPILE_MODE)
    return model, tokenizer
