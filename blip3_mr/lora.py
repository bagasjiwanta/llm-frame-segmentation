import os

from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers.modeling_utils import PreTrainedModel

from blip3_mr.config import Config
from blip3_mr.utils import isdir, log

PHI3_TARGET_MODULES = [
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
    "gate_up_proj",
    "qkv_proj",
    "gate_down_proj",
]



def load_adapter(
    model: PreTrainedModel,
    config: Config,
    task_type: str | TaskType | None = None,
    target_modules: str = "all-linear",
    model_name_or_path: str | None = None
) -> PeftModel | PeftMixedModel:
    if target_modules == "phi3":
        _target_modules = PHI3_TARGET_MODULES
    else:
        _target_modules = target_modules
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_r,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=_target_modules,
        use_rslora=config.use_rslora,
        init_lora_weights=config.init_lora_weights,
        task_type=task_type,
    )
    if model_name_or_path is None:
        if config.rank == 0:
            log("Initializing peft model using get_peft_model")

        return get_peft_model(model, lora_config)
    else:
        if config.rank == 0:
            log(f"Try loading peft model using PeftModel.from_pretrained")
        residual_model = get_peft_model(model, lora_config).unload()
        return PeftModel.from_pretrained(
            residual_model,
            model_name_or_path,
            is_trainable=True,
        )


# @TODO
def save_adapter(model: PeftModel | PeftMixedModel, path: str): ...


'''
def find_all_linear_names(model: XGenMMPerceiver, include_vision_tokenizer=False, include_lang_model=True):
    """unused"""
    assert isinstance(model, XGenMMPerceiver), f"Model must be XGenMMPerceiver, not {type(model)}"
    lora_module_names = []
    for name, module in model.named_modules():
        if "vision_encoder" in name:
            continue

        if include_vision_tokenizer and name.startswith("vision_tokenizer"):
            if isinstance(module, torch.nn.Linear):
                lora_module_names.append(name)

        if include_lang_model and name.startswith("lang_model"):
            if isinstance(module, torch.nn.Linear):
                if any(target in name for target in PHI3_TARGET_MODULES):
                    lora_module_names.append(name)

    return lora_module_names


def load_xgenmm_adapter(adapter_path: str, model: XGenMMPerceiver, config: Config) -> PeftModel:
    "Loads the whole xgenmm model (unused)"
    target_modules = find_all_linear_names(model, config.vision_tokenizer_lora, config.lang_model_lora)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_r,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
        use_rslora=True,
        init_lora_weights=INIT_LORA_WEIGHTS,
    )
    peft_model = PeftModel.from_pretrained(model=model, model_id=adapter_path, config=lora_config)
    return peft_model


def build_lora_model_from_scratch(
    config: Config,
    model: XGenMMPerceiver,
) -> PeftModel | PeftMixedModel | XGenMMPerceiver:
    """unused"""
    lora_dict = {
        "r": config.lora_r,
        "lora_alpha": config.lora_r,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "target_modules": [],
        "init_lora_weights": INIT_LORA_WEIGHTS,
        "use_rslora": True,
        "task_type": TaskType.CAUSAL_LM,
    }
    if config.resume_from_latest is not None or config.resume_from_checkpoint is not None:
        lora_dict["init_lora_weights"] = "gaussian"

    # only lang model lora
    if config.lang_model_lora and not config.vision_tokenizer_lora:
        phi3_modules = []
        for name, module in model.lang_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(target in name for target in PHI3_TARGET_MODULES):
                    phi3_modules.append(name)

        lora_dict["target_modules"] = phi3_modules
        lora_config = LoraConfig(**lora_dict)

        model.lang_model = get_peft_model(model.lang_model, lora_config)

        if config.vision_tokenizer_train:  # ensure it is not frozen when the intention is to train it
            model.vision_tokenizer.requires_grad_(True)

        # if haven't saved, then initialize first and then we no longer need to init again bcs it's slow
        if not os.path.isdir("adapters/xgenmm_langmodel/pissa_init") and config.rank == 0:
            model.lang_model.peft_config["default"].init_lora_weights = True
            model.lang_model.save_pretrained("adapters/xgenmm_langmodel/pissa_init")

        return model

    # both or only vision_tokenizer
    # if vision tokenizer is also wrapped in lora, cannot tasktype causal lm
    elif config.vision_tokenizer_lora:
        linear_names = find_all_linear_names(model, config.vision_tokenizer_lora, config.lang_model_lora)
        lora_dict["target_modules"] = linear_names
        lora_dict["task_type"] = None
        lora_config = LoraConfig(**lora_dict)

        peft_model = get_peft_model(model, lora_config)

        # unfreeze lang model if it is trained without lora since peft freezes everyhing under it
        if config.lang_model_train and not config.lang_model_lora:
            peft_model.base_model.model.lang_model.requires_grad_(True)

        if config.rank == 0:
            print("all model")
            peft_model.print_trainable_parameters()

        # if haven't saved, then initialize first and then we no longer need to init again bcs it's slow
        if not os.path.isdir("adapters/xgenmm/pissa_init") and config.rank == 0:
            peft_model.peft_config["default"].init_lora_weights = True
            peft_model.save_pretrained("adapters/xgenmm/pissa_init")

        return peft_model

    return model
'''
