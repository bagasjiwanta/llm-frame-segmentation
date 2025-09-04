import copy
import json
import os
from typing import List, Literal, TypedDict, cast

import deepspeed
import torch
import torch.distributed as dist
from peft import PeftModel
from tqdm import tqdm
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateDecoderOnlyOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.config import Config
from blip3_mr.dataset import (
    DataInfo,
    InferenceCollatorOutput,
    MomentRetrievalDataset,
    make_img_normalizer,
    make_img_resizer,
    val_batch_to_device,
)
from blip3_mr.eval_mr import eval_submission
from blip3_mr.eval_utils import (
    beam_search_to_scores,
    get_sorted_moments_from_thresholds,
    greedy_to_scores,
    process_predictions_list_sorted_moments,
)
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.utils import display_metrics_table, isdir, isfile, json_dumps, list_dict_to_jsonl, log

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "amp_bf16": torch.bfloat16,
    "amp_fp16": torch.float16,
}


class EvalDataDict(TypedDict):
    input_ids: torch.Tensor
    duration: int
    qid: int
    vid: str
    preds: str
    pred_relevant_windows: List[List[int]]


def calc_val_steps(num_epochs: int, num_micro_batch_in_epoch: int, num_val_per_epoch: int) -> list[int]:
    min_val_steps = num_micro_batch_in_epoch // num_val_per_epoch
    _val_counter = torch.arange(num_epochs)[:, None].expand(num_epochs, num_val_per_epoch)
    _val_counter = _val_counter * num_micro_batch_in_epoch
    _val_steps = torch.arange(1, num_val_per_epoch + 1) * min_val_steps
    _val_steps = _val_counter + _val_steps - 1

    # handle cases where batches are not divisible by nval
    _val_steps[:, -1] = (torch.arange(1, num_epochs + 1) * num_micro_batch_in_epoch) - 1

    val_steps = _val_steps.flatten().tolist()
    return val_steps


class ValidateReturnType(TypedDict):
    metrics: dict[str, float]
    predictions: list[dict]
    ground_truths: list[dict]


def save_val_result_to_dirs(
    output_dir: str | list[str], result: ValidateReturnType, filename="hl_val_submission.jsonl"
):
    if isinstance(output_dir, str):
        output_dir = [output_dir]

    for dirname in output_dir:
        dirname = os.path.join(os.getcwd(), dirname)
        if not isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, filename)
        predictions = json_dumps(result["predictions"])
        ground_truths = json_dumps(result["ground_truths"])
        metrics = json_dumps(result["metrics"])
        dir_pred, dir_metric, dir_gts = (
            os.path.join(dirname, "hl_val_predictions.json"),
            os.path.join(dirname, "hl_val_ground_truths.json"),
            os.path.join(dirname, "hl_val_metrics.json"),
        )
        with open(dir_pred, "w") as f_out:
            f_out.write(predictions)
        with open(dir_metric, "w") as f_out:
            f_out.write(ground_truths)
        with open(dir_gts, "w") as f_out:
            f_out.write(metrics)

        for p in result["predictions"]:
            if "score" in p:
                p.pop("score")
            if "preds" in p:
                p.pop("preds")

        vid_data = {}
        for g in result["ground_truths"]:
            vid_data[g["qid"]] = g["vid"]

        for p in result["predictions"]:
            p["vid"] = vid_data[p["qid"]]
        output_jsonl = list_dict_to_jsonl(result["predictions"])

        with open(filename, "w") as f_out:
            f_out.write(output_jsonl)

    if os.environ.get("RANK", 0) == 0:
        print(f"Val outputs are saved to {output_dir}")


def validate_one_epoch_v2(
    config: Config,
    model: deepspeed.DeepSpeedEngine | PreTrainedModel | PeftModel | XGenMMPerceiver,
    dataset: DataInfo,
    max_steps: int = 0,
) -> ValidateReturnType:
    """
    Runs a full validation loop for one epoch on the provided dataset and then display the metrics.

    Arguments:
        config (Config): The command-line arguments and configuration for the run.
        model (deepspeed.DeepSpeedEngine | PreTrainedModel | PeftModel ): The model to be evaluated.
        dataset (DataInfo): A container for the validation dataloader and dataset-specific info.
        max_steps (int): number to terminate the validation loop earlier (for sanity checking)

    Returns:
        output (ValidateReturnType): A dict containing
            - metric: dict of metrics
            - predictions: list of prediction dict
            - ground_truths: list of ground_truths dict
    """
    vrb = config.extra_verbose
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    tokenizer: PreTrainedTokenizer = dataset.tokenizer
    num_batches = len(dataset.dataloader)
    num_frames = dataset.num_frames

    img_resizer = make_img_resizer(device)
    img_normalizer = make_img_normalizer("max-autotune", device)

    model.eval()
    total_eval_steps = max_steps if max_steps > 0 else num_batches

    predictions = []
    invalid_predictions = []

    moment_dataset: MomentRetrievalDataset = dataset.dataloader.dataset
    ground_truths = moment_dataset.get_val_qvh()
    # print(len(ground_truths))
    iterator = tqdm(
        enumerate(dataset.dataloader),
        disable=config.rank != 0,
        total=total_eval_steps,
        initial=0,
        ncols=120,
        desc=f"Run validation",
    )
    for step_num, batch in iterator:
        batch: InferenceCollatorOutput
        if step_num >= total_eval_steps:
            break
        verbose = step_num <= 1 and vrb and config.rank == 0

        batch_size = batch["input_ids"].size(0)

        # move batch to device
        images, input_ids, attention_mask = val_batch_to_device(batch, device)

        # resize each image so that it becomes 384, 384, the dimension is now [Batch], Frame, Channel, Height, Width
        images = [img_resizer(frames) for frames in images]
        # stack to remove the list
        images = torch.stack(images, dim=0)
        # normalize the image in a single process
        images = img_normalizer(images)
        # create 2 new dimension, shape is Batch, Frame, 1, 1, Channel, Height, Width
        images = images.unsqueeze(2).unsqueeze(2)
        # unbind to convert first two shape to list, final shape is [Batch], [Frame], 1, 1, Channel, Height, Width (xgen-mm expects these 2 extra dims)
        images = [list(torch.unbind(image, dim=0)) for image in images]
        if config.rank == 0 and step_num == 0:
            tqdm.write("Dataloading ok")
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=dtype_map.get(config.training_precision, torch.bfloat16),
                cache_enabled=True,
                enabled=config.training_precision.startswith("amp") and not config.deepspeed,
            ):
                generation_output = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    image_size=batch["image_size"],
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=int(dataset.num_frames * 2),  # cheaper generation
                    num_beams=config.num_val_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                )  # type: ignore

        batch_size = input_ids.size(0)
        generated_text = None
        generation_output = cast(GenerateDecoderOnlyOutput, generation_output)  # for typing

        # grab the sequence and define threshold = 0.5
        generated_text = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        # remove the <|end|> token and then truncate so len(text) == num_frames in each batch
        generated_text = [g.split("<|end|>")[0][:num_frames] for g in generated_text]
        if verbose:
            tqdm.write(f"Generated text[:2] = {generated_text[:2]}")

        if config.num_val_beams > 1 and isinstance(generation_output, GenerateBeamDecoderOnlyOutput):
            scores = beam_search_to_scores(
                generation_output, dataset.token_zero, dataset.token_one, dataset.num_frames
            )
        else:
            scores = greedy_to_scores(
                generation_output.scores, dataset.token_zero, dataset.token_one, dataset.num_frames
            )

        if verbose:
            tqdm.write(f"Scores[:2] = {scores[:2]}")

        del generation_output

        for b in range(batch_size):
            pred_dict = {
                "qid": int(batch["qids"][b]),
                "duration": round(batch["durations"][b]),
                "score": scores[b].tolist(),
                "preds": generated_text[b],
            }
            if all(pred == "0" for pred in generated_text[b]) or any(
                pred not in ("1", "0") for pred in generated_text[b]
            ):
                invalid_predictions.append(pred_dict)
                pred_dict_copy = copy.deepcopy(pred_dict)
                pred_dict_copy["preds"] = "".join(["1" for _ in range(num_frames)])
                predictions.append(pred_dict_copy)
            else:
                predictions.append(pred_dict)

    if config.num_val_samples != 0:
        ground_truths = [ground_truths[k["qid"]] for k in predictions]
    else:
        ground_truths = list(ground_truths.values())

    verbose = vrb and config.rank == 0

    if dist.is_initialized():
        dist.barrier()

    process_predictions_list_sorted_moments(predictions)
    process_predictions_list_sorted_moments(invalid_predictions)

    valid_ratio = (len(predictions) - len(invalid_predictions)) / len(predictions)
    if rank == 0:
        print(f"Valid prediction ratio: {valid_ratio * 100:.1f}%")
        print(f"Number of invalid predictions: {len(invalid_predictions)}")

    if verbose:
        display_some_predictions(predictions, invalid_predictions, ground_truths)

    num_samples = len(dataset.dataloader.dataset)
    sample_ratio = len(ground_truths) / num_samples if world_size > 1 else 1.0

    metric_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * valid_ratio * sample_ratio
    try:
        _metric = eval_submission(predictions, ground_truths, False, False)
        _brief = _metric["brief"]
        metric_tensor = (
            torch.tensor(
                [
                    _brief["MR-full-mAP"],
                    _brief["MR-full-mAP@0.5"],
                    _brief["MR-full-mAP@0.75"],
                    _brief["MR-full-R1@0.5"],
                    _brief["MR-full-R1@0.7"],
                    _brief["HL-min-VeryGood-mAP"],
                    _brief["HL-min-VeryGood-Hit1"],
                    1,
                ]
            )
            * valid_ratio
            * sample_ratio
        )
    except Exception as e:
        print(f"Error evaluating submission: \n{e}\nAll metrics are set to 0.0 except the valid ratio")

    if world_size > 1:
        metric_tensor = metric_tensor.to(device)
        dist.all_reduce(metric_tensor, dist.ReduceOp.SUM, async_op=False)

    metric = {
        "mAP": metric_tensor[0].item(),
        "mAP-at-0.5": metric_tensor[1].item(),
        "mAP-at-0.75": metric_tensor[2].item(),
        "R1-at-0.5": metric_tensor[3].item(),
        "R1-at-0.7": metric_tensor[4].item(),
        "HL-min-VeryGood-mAP": metric_tensor[5].item(),
        "HL-min-VeryGood-Hit1": metric_tensor[6].item(),
        "member": metric_tensor[7].item(),  # valid member, already in wandb,
    }

    if rank == 0:
        print("\nValidation Metrics:")
        print(f"\tNum samples: {num_samples}")
        for k, v in metric.items():
            print(f"\t{k}: {v:.04f}")

    return {
        "metrics": metric,
        "ground_truths": ground_truths,
        "predictions": predictions,
        "invalid_predictions": invalid_predictions,
    }


def display_some_predictions(predictions: list, invalid_predictions: list, ground_truths: list):
    predictions_dict = {p["qid"]: p for p in predictions}
    inv_predictions_dict = {p["qid"]: p for p in invalid_predictions}
    gts_dict = {g["qid"]: g for g in ground_truths}

    max_display = min(16, len(predictions))

    # valid
    display_keys = list(predictions_dict.keys())[:max_display]
    displayed_preds = [predictions_dict[k] for k in display_keys]
    displayed_gts = [gts_dict[k] for k in display_keys]
    print(f"\nPredictions and ground truths:")
    for i, (pred, label) in enumerate(zip(displayed_preds, displayed_gts)):
        print(
            f"Qid: {pred['qid']}, Pred string: {pred['preds']}, Pred: {pred['pred_relevant_windows']}, Label: {label['relevant_windows']}"
        )

    # invalid
    display_keys = list(inv_predictions_dict.keys())[:max_display]
    displayed_inv_preds = [inv_predictions_dict[k] for k in display_keys]
    displayed_gts = [gts_dict[k] for k in display_keys]
    print(f"\nInvalid predictions and ground truths:")
    for i, (pred, label) in enumerate(zip(displayed_inv_preds, displayed_gts)):
        print(
            f"Qid: {pred['qid']}, Pred string: {pred['preds']}, Pred: {pred['pred_relevant_windows']}, Label: {label['relevant_windows']}"
        )
