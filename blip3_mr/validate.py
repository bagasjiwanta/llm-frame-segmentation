import os
from typing import List, TypedDict

import deepspeed
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers.modeling_utils import PreTrainedModel

from blip3_mr.config import Config
from blip3_mr.dataset import (
    DataInfo,
    MomentRetrievalDataset,
)
from blip3_mr.eval_mr import eval_submission
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.test import test_one_epoch
from blip3_mr.utils import isdir, json_dumps, list_dict_to_jsonl, print_dict

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
    invalid_predictions: list[dict]


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


def validate_one_epoch(
    config: Config,
    model: deepspeed.DeepSpeedEngine | PreTrainedModel | PeftModel | XGenMMPerceiver,
    dataset: DataInfo,
    max_iter: int = -1,
    do_save: bool = False,
    output_dir: str | None = None,
) -> ValidateReturnType:
    """
    Runs a full validation loop for one epoch on the provided dataset and then display the metrics.

    Arguments:
        config (Config): The command-line arguments and configuration for the run.
        model (deepspeed.DeepSpeedEngine | PreTrainedModel | PeftModel ): The model to be evaluated.
        dataset (DataInfo): A container for the validation dataloader and dataset-specific info.
        max_steps (int): number to terminate the validation loop earlier (for sanity checking)
        do_save (bool): if True, save the results to a dir
    Returns:
        output (ValidateReturnType): A dict containing
            - metric: dict of metrics
            - predictions: list of prediction dict
            - ground_truths: list of ground_truths dict
    """
    predictions, invalid_predictions = test_one_epoch(
        model=model,  # type: ignore
        dataloader=dataset.dataloader,
        precision=config.training_precision,
        tokenizer=dataset.tokenizer,
        do_save=False,
        max_iter=max_iter,
        generation_kwargs={"num_beams": config.num_val_beams},
        num_frames=dataset.num_frames,
    )

    moment_dataset: MomentRetrievalDataset = dataset.dataloader.dataset
    ground_truths = moment_dataset.get_val_qvh()
    ground_truths = [ground_truths[k["qid"]] for k in predictions]

    valid_ratio = (len(predictions) - len(invalid_predictions)) / len(predictions)

    if config.rank == 0:
        display_some_predictions(predictions, invalid_predictions, ground_truths)

    # samples is length except when set
    num_samples = len(moment_dataset) if config.num_val_samples != 0 else config.num_val_samples
    num_samples = (max_iter * config.val_batch_size * config.world_size) if max_iter > 0 else num_samples
    sample_ratio = len(ground_truths) / num_samples if config.world_size > 1 else 1.0

    metric_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, valid_ratio * sample_ratio])
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
            * valid_ratio  # eval_submission discards mismatching keys
            * sample_ratio  # how many sample this gpu holds
        )
    except Exception as e:
        print(f"Error evaluating submission: \n{e}\nAll metrics are set to 0.0 except the valid ratio")

    if config.world_size > 1:
        metric_tensor = metric_tensor.to(torch.device(f"cuda:{config.rank}"))
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

    if config.rank == 0:
        print("\nValidation Metrics:")
        print(f"\tNum samples: {num_samples}")
        print_dict(metric)

    val_results: ValidateReturnType = {
        "metrics": metric,
        "ground_truths": ground_truths,
        "predictions": predictions,
        "invalid_predictions": invalid_predictions,
    }

    if do_save and isdir(output_dir) and config.rank == 0:
        save_val_result_to_dirs([output_dir], val_results)

    return val_results


def display_some_predictions(predictions: list, invalid_predictions: list, ground_truths: list):
    MAX_DISPLAY = 16

    predictions_dict = {p["qid"]: p for p in predictions}
    inv_predictions_dict = {p["qid"]: p for p in invalid_predictions}
    gts_dict = {g["qid"]: g for g in ground_truths}

    max_display = min(MAX_DISPLAY, len(predictions))

    # valid
    display_keys = list(predictions_dict.keys())[:max_display]
    displayed_preds = [predictions_dict[k] for k in display_keys]
    displayed_gts = [gts_dict[k] for k in display_keys]
    print("\nPredictions and ground truths:")
    for i, (pred, label) in enumerate(zip(displayed_preds, displayed_gts)):
        print(
            f"Qid: {pred['qid']}, Pred string: {pred['preds']}, Pred: {pred['pred_relevant_windows']}, Label: {label['relevant_windows']}"
        )

    # invalid
    display_keys = list(inv_predictions_dict.keys())[:max_display]
    displayed_inv_preds = [inv_predictions_dict[k] for k in display_keys]
    displayed_gts = [gts_dict[k] for k in display_keys]
    print("\nInvalid predictions and ground truths:")
    for i, (pred, label) in enumerate(zip(displayed_inv_preds, displayed_gts)):
        print(
            f"Qid: {pred['qid']}, Pred string: {pred['preds']}, Pred: {pred['pred_relevant_windows']}, Label: {label['relevant_windows']}"
        )
