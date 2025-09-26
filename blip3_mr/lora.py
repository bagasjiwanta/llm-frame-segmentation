from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers.modeling_utils import PreTrainedModel

from blip3_mr.config import Config
from blip3_mr.utils import log

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
    model_name_or_path: str | None = None,
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
