from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from transformers.modeling_utils import PreTrainedModel

from blip3_mr.config import Config
from blip3_mr.utils import log


def load_adapter(
    model: PreTrainedModel,
    config: Config,
    task_type: str | TaskType | None = None,
    target_modules: str = "all-linear",
    model_name_or_path: str | None = None,
) -> PeftModel | PeftMixedModel:
    if model_name_or_path is None:
        if config.rank == 0:
            log("Initializing peft model using get_peft_model")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_r,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=target_modules,
            use_rslora=config.use_rslora,
            init_lora_weights=config.init_lora_weights,
            task_type=task_type,
        )

        return get_peft_model(model, lora_config)
    else:
        log(f"Try loading peft model using PeftModel.from_pretrained")
        return PeftModel.from_pretrained(
            model,
            model_name_or_path,
            subfolder="pissa_init",
            is_trainable=True,
        )
