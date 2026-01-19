import time
from typing import Generator

import torch
from deepspeed import DeepSpeedEngine
from tqdm import tqdm

from blip3_mr.config import Config
from blip3_mr.dataset import (
    DataInfo,
    TrainCollatorOutput,
    make_img_normalizer,
    make_img_resizer,
    process_images,
    train_batch_to_device,
)
from blip3_mr.losses import (
    extract_binary_mask_from_logits,
    setup_finetune_losses,
)
from blip3_mr.utils import (
    TrainingMeters,
)


def deepspeed_finetune_one_epoch_generator(
    config: Config,
    resume_from_step: int,
    model: DeepSpeedEngine,
    epoch: int,
    datainfo: DataInfo,
    meters: TrainingMeters,
    device: torch.device,
) -> Generator[tuple[int, int, int], None, None]:
    """
    Helper function for running one epoch of training.
    Handles logging, calling forward, backward, gradient clipping, and optimizer step.

    Arguments:
        config (Config): arguments from command line
        resume_from_step (int): step number to resume, must be from 0 to len(dataloader) - 1 since this function is operated epoch based
        model (deepspeed.DeepSpeedEngine): Deepspeed Engine
        epoch (int): epoch number
        datainfo (DataInfo): train dataset
        meters (TrainingMeter): training meters

    Returns:
        dict[str, float]: dictionary of metrics
    """
    num_frame = datainfo.num_frames
    dataloader = datainfo.dataloader
    num_class = 2

    img_resizer = make_img_resizer(device)
    img_normalizer = make_img_normalizer("max-autotune", device)

    generalized_dice_l, tversky_l, binary_cross_entropy_l = setup_finetune_losses(config, device, num_frame)

    datainfo.set_epoch(epoch)

    end_time = time.time()
    for local_step, batch in enumerate(dataloader):
        if config.rank == 0 and local_step == 0:
            tqdm.write("Dataloading ok")
        model.train()
        batch: TrainCollatorOutput
        batch_size = batch["input_ids"].size(0)
        if local_step < resume_from_step:
            continue
        true_step = local_step + (epoch * len(dataloader))

        meters.data_time.update((time.time() - end_time) * 1000)
        meters.num_pos.update(float(batch["answers"][:, :, -1].sum().item() / (batch_size * num_frame)))

        images, input_ids, attention_mask, labels, answers = train_batch_to_device(batch, device)
        if config.use_local_model:
            images = process_images(images, img_resizer, img_normalizer)

        meters.num_tokens.update(attention_mask.sum().item() / 1000)

        lang_labels = labels if config.loss_ce_weight > 0.0 else None
        # tqdm.write("dataloading ok")
        output = model(
            vision_x=images,
            image_size=batch["image_size"],
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=lang_labels,
        )
        logits = output.logits

        moment_logits = extract_binary_mask_from_logits(logits, input_ids, num_frame, num_class, datainfo=datainfo)

        loss = 0.0

        if config.loss_bce_weight > 0.0:
            binary_mask = moment_logits[:, :, 1] - moment_logits[:, :, 0]  # B,F
            binary_answ = answers[:, :, 1]  # probs from dataset, originally B,F,2
            bce_loss = binary_cross_entropy_l(binary_mask, binary_answ)
            if meters.bce_loss is not None:
                meters.bce_loss.update(bce_loss.item())
            loss += bce_loss * config.loss_bce_weight

        if config.loss_gd_weight > 0.0:
            gd_loss = generalized_dice_l(moment_logits, answers)
            if meters.gd_loss is not None:
                meters.gd_loss.update(gd_loss.item())
            loss += gd_loss * config.loss_gd_weight

        if config.loss_tvl_weight > 0.0:
            tv_loss = tversky_l(moment_logits, answers)
            if meters.tv_loss is not None:
                meters.tv_loss.update(tv_loss.item())
            loss += tv_loss * config.loss_tvl_weight

        if config.loss_ce_weight > 0.0:
            ce_loss = output.loss
            if meters.ce_loss is not None:
                meters.ce_loss.update(ce_loss.item())
            loss += ce_loss * config.loss_ce_weight

        model.backward(loss)
        model.step()

        meters.step_time.update(time.time() - end_time)
        end_time = time.time()
        meters.lr.update(model.optimizer.get_lr())  # type: ignore

        yield model.global_steps, true_step, local_step
