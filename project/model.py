import copy
from collections import UserDict
from dataclasses import asdict
from typing import Literal, Optional, TypedDict, Union, cast

import lightning as L
import torch
from peft import LoraConfig, PeftMixedModel, PeftModel, TaskType, get_peft_model
from torch import FloatTensor, LongTensor, Tensor, nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerateDecoderOnlyOutput
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.models.phi3 import Phi3ForCausalLM
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from project.base_models.blip3 import XGenMMModelForConditionalGeneration
from project.config import Config
from project.dataset import InferenceCollatorOutput, TrainCollatorOutput
from project.eval_utils import beam_search_to_scores, greedy_to_scores
from project.losses import GeneralizedDiceLoss, TverskyLoss
from project.utils import log


class GenerativePreTrainedModel(PreTrainedModel, GenerationMixin):
    """For type hint"""

    pass


def load_adapter(
    model: PreTrainedModel,
    config: Config,
    task_type: str | TaskType | None = None,
    target_modules: str | list[str] | None = "all-linear",
    model_path: str | None = None,
) -> PeftModel | PeftMixedModel:
    """Creates a new `PeftModel` or load from `model_path` if `model_path` is not None"""

    if model_path is None:
        log(
            f"Creating PeftModel of {model._get_name()} and automatically saving it to "
            "checkpoint dir if present in config. "
            "This might take up hours and it's better to preprocess the model first for reuse."
        )
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
        log(f"Loading PeftModel from {model_path}")
        return PeftModel.from_pretrained(model, model_path, is_trainable=True)


class MyModule(L.LightningModule):
    num_frames: int = 25
    model: GenerativePreTrainedModel | XGenMMModelForConditionalGeneration
    tokenizer: PreTrainedTokenizer
    criterions: list[nn.Module]
    model_cfg: PretrainedConfig
    model_code: Literal["blip3", "qwen3-vl", "internvl3"]
    peft_model: PeftModel | PeftMixedModel | None

    def __init__(self, config: Config):
        super().__init__()
        # configs
        self.save_hyperparameters(**asdict(config))
        self.cfg = config

        # init model
        self.model_cfg = AutoConfig.from_pretrained(self.cfg.model_name_or_path)
        self.model = AutoModelForImageTextToText.from_pretrained(self.cfg.model_name_or_path, config=self.model_config)

        # type model
        name = self.model.__class__.__name__.lower()
        if "xgenmm" in name and "video" not in name:
            self.model_code = "blip3"
        else:
            self.model_code = "blip3"
        log(f"model_code is {self.model_code}")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, config=self.model_config)
        if self.model_code == "blip3":
            self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.infer_tokens_from_tokenizer()

        # losses & predictions
        self.init_losses(config)

        self.predictions = []
        self.invalid_predictions = []

        # self.test_predictions = []

    def infer_tokens_from_tokenizer(self):
        tokenizer = self.tokenizer

        token_0 = tokenizer.convert_tokens_to_ids("0")
        self.token_0 = token_0[-1] if isinstance(token_0, list) else token_0

        token_1 = tokenizer.convert_tokens_to_ids("1")
        self.token_1 = token_1[-1] if isinstance(token_1, list) else token_1

        start_offset = 0
        if self.model_code == "blip3":
            token_start = tokenizer.convert_tokens_to_ids("<|assistant|>")
            self.token_start = token_start[-1] if isinstance(token_start, list) else token_start
        # elif self.model_code == 'qwen3-vl':
        else:
            token_start = tokenizer.convert_tokens_to_ids("<|assistant|>")
            self.token_start = token_start[-1] if isinstance(token_start, list) else token_start

        self.start_offset = start_offset

    def wrap_model_in_peft(self):
        model = self.model
        if self.model_code == "blip3":
            self.peft_model = load_adapter(
                model.vlm.lang_model,
                self.cfg,
                TaskType.CAUSAL_LM,
                target_modules="all-linear",
                model_path=self.cfg.adapter_path,
            )

    def init_losses(self, config: Config):
        bce_crit = nn.BCEWithLogitsLoss(pos_weight=torch.ones((1,)))
        gd_crit = GeneralizedDiceLoss(soft_labels=config.soft_loss)
        tv_crit = TverskyLoss(soft_labels=config.soft_loss, beta=config.tv_beta, alpha=1.0 - config.tv_beta)

        self.criterions = [bce_crit, gd_crit, tv_crit]
        self.crit_names = ["bce_loss", "gd_loss", "tv_loss"]
        self.crit_weight = [config.loss_bce_weight, config.loss_gd_weight, config.loss_tv_weight]

    def extract_binary_mask(self, logits: FloatTensor, input_ids: LongTensor) -> Tensor:
        B = logits.size(0)
        argmax_dim = 1 if B > 1 else 0
        num_class = 2

        start_mask = (input_ids == self.token_start).int().argmax(dim=argmax_dim).view(B, 1)
        start_mask += self.start_offset

        start_offsets = torch.arange(self.num_frames).unsqueeze(0).to(logits.device, non_blocking=True)

        batch_idxs = torch.arange(B).view(B, 1, 1).expand(B, self.num_frames, num_class)
        frame_idxs = (start_offsets + start_mask).view(B, self.num_frames, 1).expand(B, self.num_frames, num_class)
        probs_idxs = torch.tensor([self.token_0, self.token_1]).expand(B, self.num_frames, num_class)

        moment_logits = logits[batch_idxs, frame_idxs, probs_idxs]  # (Batch, Frame, 2)

        return moment_logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.model(**batch)
        self.log(name="train/ce_loss", value=loss.item())

        moment_logits = self.extract_binary_mask(logits=logits, input_ids=batch.input_ids)

        log_freq = max(self.cfg.gradient_accumulation_steps, 2)
        for idx, crit in enumerate(self.criterions):
            crit_loss = crit(moment_logits)
            loss += crit_loss * self.crit_weight[idx]
            self.log(
                name="train/" + self.crit_names[idx],
                value=crit_loss.item(),
                on_step=(batch_idx + 1) % log_freq == 0,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        self.predict_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        cfg = self.cfg
        output = self.model.generate(
            pixel_values=batch.pixel_values,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=self.num_frames * 2,
            num_beams=self.cfg.num_val_beams,
        )
        output = cast(GenerateDecoderOnlyOutput, output)
        text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        text = [text.split("<|end|>")[0][: self.num_frames] for text in text]

        if cfg.num_val_beams > 1 and isinstance(output, GenerateBeamDecoderOnlyOutput):
            scores = beam_search_to_scores(output, self.token_0, self.token_1, self.num_frames)
        else:
            scores = greedy_to_scores(
                output.scores,
                self.token_0,
                self.token_1,
                self.num_frames,
            )
        del output

        for b in range(cfg.train_batch_size):
            pred_dict = {
                "qid": int(batch["qids"][b]),
                "duration": round(batch["durations"][b]),
                "score": scores[b].tolist(),
                "preds": text[b],
            }
            if all(pred == "0" for pred in text[b]) or any(pred not in ("1", "0") for pred in text[b]):
                self.invalid_predictions.append(pred_dict)
                pred_dict_copy = copy.deepcopy(pred_dict)
                pred_dict_copy["preds"] = "".join(["1" for _ in range(self.num_frames)])
                self.predictions.append(pred_dict_copy)
            else:
                self.predictions.append(pred_dict)

    def on_validation_epoch_end(self) -> None:
        
        self.predictions.clear()
        self.invalid_predictions.clear()

    def on_predict_epoch_end(self) -> None:
        
        self.predictions.clear()
        self.invalid_predictions.clear()

    def configure_optimizers(self):  # type: ignore
        cfg = self.cfg
        if cfg.deepspeed:  # deepspeed is taken care of by lightning
            return super().configure_optimizers()
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                fused=True,
            )
            lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                cfg.warmup_steps,
                cfg.num_train_samples,
                num_cycles=2.0,
                min_lr_rate=0.05,
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
