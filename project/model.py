import lightning as L
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from project.config import Config
from project.losses import GeneralizedDiceLoss, TverskyLoss


class BaseModelForVTG(nn.Module):
    """Abstract Class"""

    def __init__(self):
        pass

    def process_images(self, images):
        pass

    def forward(self, *params, **kwargs):
        pass

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
        tv_crit = TverskyLoss(soft_labels=config.soft_loss, beta=config.tv_beta, alpha=1.0 - config.tv_alpha)

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


class BLIP3ForVTGConfig(PretrainedConfig):
    def __init__(self, model_name: str, token_per_image: int = 32):
        self.model_name = model_name
        self.token_per_image = token_per_image


class BLIP3ForVTG(PreTrainedModel):
    config_class = BLIP3ForVTGConfig

    def __init__(self, config: BLIP3ForVTGConfig):
        super().__init__(config)
