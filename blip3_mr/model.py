from typing import cast

import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.config import Config, get_deepspeed_config_from_config
from blip3_mr.lora import (
    load_adapter,
)
from blip3_mr.open_flamingo.src.factory import create_model_and_tokenizer
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.utils import (
    find_and_load_checkpoint_deepspeed,
    isdir,
    isfile,
    load_pretrained_state_dict,
    log,
)

COMPILE_MODE = "default"


def load_model(config: Config) -> tuple[XGenMMPerceiver, PreTrainedTokenizer]:
    if config.use_local_model:
        lang_model_pretrained = "microsoft/Phi-3-mini-4k-instruct"
        if isdir(config.lang_model_pretrained) or isinstance(config.lang_model_pretrained, str):
            lang_model_pretrained = config.lang_model_pretrained
        model, tokenizer = create_model_and_tokenizer(
            gradient_checkpointing=config.gradient_checkpointing,
            pretrained=config.base_model_name_or_path,
            lang_model_path=lang_model_pretrained,
        )
    else:
        hf_model = AutoModelForVision2Seq.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer = hf_model.update_special_tokens(tokenizer)
        model = hf_model.vlm
        if config.rank == 0:
            print(hf_model.config)
        model = cast(XGenMMPerceiver, model)

    if config.do_train:
        model.set_trainable()
    else:
        model.requires_grad_(False)

    if config.rank == 0:
        print(f"Total parameters:\n{model.num_params_per_module}")

    if isfile(config.vision_tokenizer_pretrained):
        log(f"Loading vision_tokenizer from {config.vision_tokenizer_pretrained}")
        load_pretrained_state_dict(
            model.vision_tokenizer,
            config.vision_tokenizer_pretrained,  # type: ignore
            bad_key="vision_tokenizer.",
        )

    if isfile(config.lang_model_pretrained):
        log(f"Loading lang_model from {config.lang_model_pretrained}")
        load_pretrained_state_dict(model.lang_model, config.lang_model_pretrained, bad_key="lang_model.")  # type: ignore

    return model, tokenizer


def wrap_model_in_lora(config: Config, model: XGenMMPerceiver, tokenizer: PreTrainedTokenizer):
    if not config.lang_model_lora:
        return

    if isinstance(config.lang_model_adapter, str):
        model.lang_model = AutoModelForCausalLM.from_pretrained(
            config.lang_model_adapter,
            dtype=torch.bfloat16,
        )
        peft_model = load_adapter(model.lang_model, config, model_name_or_path=config.lang_model_adapter)
        log(f"Loaded lang_model adapter from {config.lang_model_adapter} with config:")
    else:
        peft_model = load_adapter(
            model.lang_model,
            config,
            task_type="CAUSAL_LM",
        )
        log(f"Created lang_model adapter with config:")

    peft_model.to(torch.bfloat16)
    model.lang_model = peft_model
    if config.rank == 0:
        print(peft_model.peft_config)
        log("Lang model trainable parameters:")
        peft_model.print_trainable_parameters()


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

    if ckpt_ok and config.lora and config.deepspeed_from_universal and False:
        if config.lang_model_lora:
            log("Reloading lora model for universal checkp0oint")
            model.lang_model = load_adapter(model.lang_model, config)
    return ckpt_dir, resume_from_step, resume_from_epoch, deepspeed_model


def create_model(config: Config) -> tuple[XGenMMPerceiver, PreTrainedTokenizer]:
    model, tokenizer = load_model(config)
    wrap_model_in_lora(config, model, tokenizer)

    return model, tokenizer
