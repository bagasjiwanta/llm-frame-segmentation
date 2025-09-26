import copy
import json
import os
from typing import Literal, cast

import torch
import torch.distributed as dist
from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.dataset import (
    InferenceCollatorOutput,
    make_img_normalizer,
    make_img_resizer,
    process_images,
    val_batch_to_device,
)
from blip3_mr.eval_utils import (
    beam_search_to_scores,
    greedy_to_scores,
    inplace_pred_string_to_relevant_windows,
    # process_predictions_list_sorted_moments,
)
from blip3_mr.open_flamingo.src.xgenmm import XGenMMPerceiver
from blip3_mr.utils import list_dict_to_jsonl

dtype_map = {
    "bf16": torch.bfloat16,
    "amp_bf16": torch.bfloat16,
    "fp16": torch.float16,
    "amp_fp16": torch.float16,
    "fp32": torch.float32,
}


def infer_tokens_from_tokenizer(tokenizer: PreTrainedTokenizer):
    token_zero = tokenizer.convert_tokens_to_ids("0")
    token_zero = token_zero[-1] if isinstance(token_zero, list) else token_zero

    token_one = tokenizer.convert_tokens_to_ids("1")
    token_one = token_one[-1] if isinstance(token_one, list) else token_one

    # token_assistant = tokenizer.convert_tokens_to_ids("<|assistant|>")
    # token_assistant = token_assistant[-1] if isinstance(token_assistant, list) else token_assistant

    # return token_zero, token_one, token_assistant
    return token_zero, token_one


def test_one_epoch(
    model: XGenMMPerceiver,
    dataloader: DataLoader | None,
    precision: Literal["bf16", "fp32", "amp_bf16", "fp16", "amp_fp16"],
    generation_kwargs: dict,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | list[str] = "runs/latest",
    filename: str = "hl_test_submission.jsonl",
    do_save: bool = True,
    max_iter: int = -1,
):
    """
    Do inference on dataloader. This can be used without other parts of the repo.

    filename and output_dir is not needed if not do_save
    """
    assert dataloader is not None

    rank = os.environ.get("RANK", 0)
    if dist.is_initialized():
        rank = dist.get_rank()

    device = torch.device(f"cuda:{rank}")

    img_resizer, img_normalizer = make_img_resizer(device), make_img_normalizer(device=device)

    if not precision.startswith("amp"):
        model.to(dtype=dtype_map.get(precision))

    model.eval()

    predictions = []
    invalid_predictions = []

    token_zero, token_one = infer_tokens_from_tokenizer(tokenizer)

    iterator = tqdm(enumerate(dataloader), disable=rank != 0, ncols=120, desc="Run Inference", total=len(dataloader))
    for step, batch in iterator:
        if max_iter > 0 and (step + 1) > max_iter:
            break

        batch: InferenceCollatorOutput

        batch_size = batch["input_ids"].size(0)
        num_frames = 25

        images, input_ids, attention_mask = val_batch_to_device(batch, device)
        images = process_images(images, img_resizer, img_normalizer)

        if rank == 0 and step == 0:
            tqdm.write("Dataloading OK")
        num_beams = generation_kwargs.get("num_beams", 1)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                enabled=precision.startswith("amp") and not isinstance(model, DeepSpeedEngine),
                cache_enabled=True,
                dtype=dtype_map.get(precision),
            ):
                generation_output = model.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    image_size=batch["image_size"],
                    attention_mask=attention_mask,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=generation_kwargs.get("max_new_tokens", int(num_frames * 2)),
                    num_beams=num_beams,
                )
                generation_output = cast(GenerateDecoderOnlyOutput, generation_output)

        generated_text = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        generated_text = [text.split("<|end|>")[0][:num_frames] for text in generated_text]

        if step <= 1:
            tqdm.write(f"Generated text[:2] = {generated_text[:2]}")

        if num_beams > 1 and isinstance(generation_output, GenerateBeamDecoderOnlyOutput):
            scores = beam_search_to_scores(generation_output, token_zero, token_one, num_frames)
        else:
            scores = greedy_to_scores(
                generation_output.scores,
                token_zero,
                token_one,
                num_frames,  # type: ignore
            )

        if step <= 1:
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

    if dist.is_initialized():
        dist.barrier()

    inplace_pred_string_to_relevant_windows(predictions, num_moments=10)
    inplace_pred_string_to_relevant_windows(invalid_predictions, num_moments=10)

    data_dict = dataloader.dataset.list_data_dict
    qidvid = {d["id"]: d["vid"] for d in data_dict}

    for pred in predictions:
        pred["vid"] = qidvid[pred["qid"]]

    valid_ratio = (len(predictions) - len(invalid_predictions)) / len(predictions)
    if rank == 0:
        print(f"Valid prediction ratio: {valid_ratio * 100:.1f}%")
        print(f"Number of invalid predictions: {len(invalid_predictions)}")
        print(f"Number of valid predictions: {len(predictions)}")

    if do_save:
        submission = copy.deepcopy(predictions)
        for p in range(len(submission)):
            submission[p].pop("preds")
            submission[p].pop("score")
        save_test_result_to_dir(output_dir, submission, filename)
        save_test_result_to_dir(output_dir, predictions, "hl_test_predictions.json")
        save_test_result_to_dir(output_dir, invalid_predictions, "hl_test_invalid_predictions.json")
        print(f"Test outputs are saved to {output_dir}")
    else:
        return predictions, invalid_predictions


def save_test_result_to_dir(output_dir: str | list[str], result: list[dict], filename="hl_test_submission.jsonl"):
    if isinstance(output_dir, str):
        output_dir = [output_dir]
    out_dirs: list[str] = []
    for dirname in output_dir:
        dirname = os.path.join(os.getcwd(), dirname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        dirname = os.path.join(dirname, filename)
        if filename.endswith("jsonl"):
            output = list_dict_to_jsonl(result)
        else:
            output = json.dumps(result, indent=2)
        with open(dirname, "w") as f_out:
            f_out.write(output)
        out_dirs.append(dirname)
    return out_dirs
