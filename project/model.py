from typing import Optional, Union

import lightning as L
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.internvl.modeling_internvl import InternVLForConditionalGeneration

from project.base_models.blip3 import XGenMMConfig, XGenMMModelForConditionalGeneration
from project.config import Config
from project.losses import GeneralizedDiceLoss, TverskyLoss


class BaseModelForVTG(nn.Module):
    """Abstract Class"""

    model: PreTrainedModel

    def __init__(self):
        pass

    def process_images(self, images):
        pass

    def forward(self, *params, **kwargs):
        return self.model(*params, **kwargs)

    def extract_binary_mask(self, logits):
        pass


class MyModule(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.model: BaseModelForVTG
        self.criterions: list[nn.Module]
        self.cfg = config

        bce_crit = nn.BCEWithLogitsLoss(pos_weight=torch.ones((50,)))
        gd_crit = GeneralizedDiceLoss(soft_labels=config.soft_loss)
        tv_crit = TverskyLoss(soft_labels=config.soft_loss, beta=config.tv_beta, alpha=1.0 - config.tv_beta)

        self.criterions = [bce_crit, gd_crit, tv_crit]
        self.crit_names = ["bce", "gd", "tv"]
        self.crit_weight = [config.loss_bce_weight, config.loss_gd_weight, config.loss_tv_weight]

    def training_step(self, batch, batch_idx):
        # postprocess images (model specific code)
        processed_batch = self.model.process_images(batch)
        # forward
        loss, logits = self.model(processed_batch)
        # extract logits in the right places
        moment_logits = self.model.extract_binary_mask(logits=logits)
        # add and log every loss
        for idx, crit in enumerate(self.criterions):
            crit_loss = crit(moment_logits)
            loss += crit_loss * self.crit_weight[idx]
            self.log(
                name=self.crit_names[idx],
                value=crit_loss.item(),
                prog_bar=True,
                logger=True,
                on_step=(batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0,
                sync_dist=True,
            )
        return loss


class BLIP3_ForVTGConfig(PretrainedConfig):
    def __init__(
        self,
        model_name_or_path: str,
        num_final_vis_tokens: int | None = 32,
        vis_proj_type: str = "linear",
    ):
        """
        Args:
            model_name_or_path (str): name or path
            num_final_vis_tokens (int | None): num of vision tokens that each frame gets
            vis_proj_type (str): projection from vision tokenizer to LLM
        """
        self.model_name_or_path = model_name_or_path
        self.num_final_vis_tokens = num_final_vis_tokens
        self.vis_proj_type = vis_proj_type


class BLIP3_ForVTG(PreTrainedModel):
    config_class = BLIP3_ForVTGConfig

    def __init__(self, config: BLIP3_ForVTGConfig):
        super().__init__(config)
        self.config = config
        self.model_config = XGenMMConfig.from_pretrained(config.model_name_or_path)
        if config.num_final_vis_tokens is not None:
            self.model_config.vision_tokenizer_config.num_final_vis_tokens = config.num_final_vis_tokens

        self.model_config.vis_proj_type = self.vis_proj_type
        self.model = XGenMMModelForConditionalGeneration.from_pretrained(
            self.config.model_name_or_path, config=self.model_config
        )


class InternVL2_5_ForVTGConfig(PretrainedConfig):
    def __init__(self):
        pass


class InternVL2_5_ForVTG(PreTrainedModel):
    config_class = InternVL2_5_ForVTGConfig


class Qwen2_5VL_ForVTGConfig(PretrainedConfig):
    def __init__(self):
        pass
