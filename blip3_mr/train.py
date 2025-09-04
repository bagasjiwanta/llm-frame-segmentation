import time
from typing import Generator

import torch
from deepspeed import DeepSpeedEngine
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from blip3_mr.config import Config
from blip3_mr.dataset import (
    DataInfo,
    TrainCollatorOutput,
    make_img_normalizer,
    make_img_resizer,
    train_batch_to_device,
)
from blip3_mr.losses import (
    extract_binary_mask_from_logits,
    setup_finetune_losses,
    weighted_cross_entropy,
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
    vocab_size = len(datainfo.tokenizer)

    img_resizer = make_img_resizer(device)
    img_normalizer = make_img_normalizer("max-autotune", device)

    cross_entropy_weight, generalized_dice_l, tversky_l, binary_cross_entropy_l = setup_finetune_losses(
        config, datainfo, device, num_frame
    )

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
        images = [img_resizer(frames) for frames in images]  # [B],F,C,H,W
        images = torch.stack(images, dim=0)  # B,F,C,H,W
        images = img_normalizer(images)
        images = images.unsqueeze(2).unsqueeze(2)
        images = [
            list(torch.unbind(image, dim=0)) for image in images
        ]  # [B],[F],1,P,C,H,W  xgenmm expects this dimension

        meters.num_tokens.update(attention_mask.sum().item() / 1000)

        lang_labels = labels if (config.loss_ce_weight > 0.0 and config.ce_pos_weight == 1.0) else None
        output = model(
            vision_x=images,
            image_size=batch["image_size"],
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=lang_labels,
        )
        logits = output.logits

        moment_logits = extract_binary_mask_from_logits(
            logits, input_ids, num_frame, num_class, datainfo=datainfo
        )

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
            if config.ce_pos_weight == 1.0:
                if local_step == 0 and config.rank == 0:
                    tqdm.write("using builtin loss for lang model")
                # use builtin loss in the lang_model
                ce_loss = output.loss
                if meters.ce_loss is not None:
                    meters.ce_loss.update(ce_loss.item())
                loss += ce_loss * config.loss_ce_weight
            else:
                # use self-defined loss if the pos_weight > 1.0
                logits_trunc = logits[:, -labels.size(1) :, :]
                ce_loss = weighted_cross_entropy(
                    logits_trunc, labels, vocab_size, "mean", cross_entropy_weight
                )
                if meters.ce_loss is not None:
                    meters.ce_loss.update(ce_loss.item())
                loss += ce_loss * config.loss_ce_weight

        model.backward(loss)
        model.step()

        meters.step_time.update(time.time() - end_time)
        # torch.cuda.synchronize()  let the meters get their own time and then allreduce
        end_time = time.time()
        meters.lr.update(model.optimizer.get_lr())

        yield model.global_steps, true_step, local_step
