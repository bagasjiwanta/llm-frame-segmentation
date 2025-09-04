import argparse
import json
import os
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor

# from transformers.generation.utils import BeamSearchDecoderOnlyOutput


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class pred_datadict(TypedDict):
    preds: str
    duration: int
    score: list[float]
    pred_relevant_windows: list[list[int]] | None
    pred_saliency_scores: list[int] | None


def process_predictions_list_sorted_moments(
    list_of_predictions: list[pred_datadict], num_moments: int
) -> None:
    """
    Processes a list of prediction dictionaries, adding sorted moments
    and saliency scores to each dictionary in place.
    """
    for prediction_dict in list_of_predictions:
        get_sorted_moments_from_thresholds(prediction_dict, num_moments)


def get_sorted_moments_from_thresholds(data_dict: pred_datadict, num_moments: int) -> None:
    """
    Generates moments from multiple thresholds, combines them, removes duplicates,
    sorts them by score (descending), and adds them to 'pred_relevant_windows'.

    This function modifies the input dictionary in place.

    Parameters:
        data_dict: A single prediction dictionary containing 'score' and 'duration'.
    """
    scores = data_dict["score"]
    duration = data_dict["duration"]
    num_scores = len(scores)
    time_step = duration / num_scores

    # thresholds = np.arange(0.0, 1.0, 0.25)
    thresholds = np.array([0.5])
    all_moments_set = set()

    for threshold in thresholds:
        preds = "".join(["1" if s >= threshold else "0" for s in scores])

        in_span = False
        span_start_idx = None
        start_time = 0

        for j, val in enumerate(preds):
            if val == "1" and not in_span:
                in_span = True
                start_time = round(j * time_step)
                span_start_idx = j
            elif val == "0" and in_span:
                end_time = round(j * time_step)
                span_scores = scores[span_start_idx:j]
                if span_scores:
                    avg_score = float(sum(span_scores)) / len(span_scores)
                    # Add moment as a tuple to the set to ensure uniqueness
                    all_moments_set.add((float(start_time), float(end_time), avg_score))
                in_span = False
                span_start_idx = None

        if in_span:
            end_time = int(duration)
            span_scores = scores[span_start_idx:num_scores]
            if span_scores:
                avg_score = float(sum(span_scores)) / len(span_scores)
                all_moments_set.add((float(start_time), float(end_time), avg_score))

    unique_moments = [list(moment) for moment in all_moments_set]

    sorted_moments = sorted(unique_moments, key=lambda x: x[2], reverse=True)

    if len(sorted_moments) > num_moments:
        sorted_moments = sorted_moments[:num_moments]
    elif len(sorted_moments) < num_moments:
        remain = num_moments - len(sorted_moments)
        remaining = [[0.0, 150.0, 0.0] for i in range(remain)]
        sorted_moments.extend(remaining)

    data_dict["pred_relevant_windows"] = sorted_moments

    pred_saliency_scores = generate_2s_scores_interpolated(duration, scores)
    data_dict["pred_saliency_scores"] = pred_saliency_scores


def add_windows_to_masked_predictions(data_dict: list[pred_datadict]) -> None:
    """
    Turns a prediction string into a list of spans with scores.

    If all pred_datadict does not contain 'score', then, score will be 1.0 for every pred in preds.

    Parameters:
        data_dict: list of dictionaries, each containing "preds" (string of '0'/'1'), "duration" (int), and "score" (list of float, same length as preds)
    Returns:
        None. Modifies data_dict in place, adding "pred_relevant_windows" field containing spans as [start, end, avg_score]
    """
    num_data = len(data_dict)

    preds = [data_dict[i]["preds"] for i in range(num_data)]
    durations = [data_dict[i]["duration"] for i in range(num_data)]

    if all("score" in data_dict[i] for i in range(num_data)):
        scores_list = [data_dict[i]["score"] for i in range(num_data)]
    else:
        scores_list = [[1.0 for _ in range(len(preds[i]))] for i in range(num_data)]

    for idx, (pred, duration, scores) in enumerate(zip(preds, durations, scores_list)):
        N = len(pred)
        time_step = duration / N
        spans = []
        in_span = False
        span_start_idx = None
        start = 0
        for j, val in enumerate(pred):
            if val == "1" and not in_span:
                in_span = True
                start = round(j * time_step)
                span_start_idx = j
            elif val == "0" and in_span:
                end = round(j * time_step)  # for timestep
                # Compute average score for this span
                span_scores = scores[span_start_idx:j]
                avg_score = float(sum(span_scores)) / len(span_scores)
                spans.append([start, end, avg_score])
                in_span = False
                span_start_idx = None
        # Handle case where span goes to end
        if in_span:
            end = int(duration)
            span_scores = scores[span_start_idx:N]
            avg_score = float(sum(span_scores)) / len(span_scores)
            spans.append([start, end, avg_score])
        data_dict[idx]["pred_relevant_windows"] = spans

        pred_saliency_scores = generate_2s_scores_interpolated(duration, scores)
        data_dict[idx]["pred_saliency_scores"] = pred_saliency_scores


def generate_2s_scores_interpolated(duration: int, scores: list) -> list[int]:
    """
    Generates per 2 second scores for a video for qvhighlights submission.

    This function implements the weighted average logic for 2-second clips
    that straddle the boundaries of the original score segments.

    Args:
        duration (int): The total duration of the video in seconds.
        scores (list): A list of 25 float scores from the model.

    Returns:
        np.ndarray: An array of scores, one for each 2-second interval.
    """
    num_scores = len(scores)
    if num_scores == 0:
        return []

    segment_duration = duration / num_scores
    per_2s_scores = []

    # Iterate through the video in 2-second intervals
    for start_time_2s in range(0, duration, 2):
        end_time_2s = min(start_time_2s + 2, duration)
        start_segment_idx = int(start_time_2s / segment_duration)
        end_segment_idx = int((end_time_2s - 1e-8) / segment_duration)

        # Case 1: The whole clip is contained in 1 segment
        if start_segment_idx == end_segment_idx:
            final_score = scores[start_segment_idx]

        # Case 2: The boundary of two segments lands inside the clip
        else:
            # Time of the boundary between the two segments
            boundary_time = (start_segment_idx + 1) * segment_duration
            # duration of the 2s clip spent in the first segment
            overlap1 = boundary_time - start_time_2s
            # duration of the 2s clip spent in the second segment
            overlap2 = end_time_2s - boundary_time

            # weighted avg
            score1 = scores[start_segment_idx]
            score2 = scores[end_segment_idx]
            total_interval_len = end_time_2s - start_time_2s
            final_score = (overlap1 * score1 + overlap2 * score2) / total_interval_len

        per_2s_scores.append(final_score)

    return per_2s_scores


def test_clip_score_generation():
    video_duration = 138
    model_scores = [0.8, 0.7, 0.6, 0.5, 0.4] * 5  # A list of 25 scores

    two_second_scores = generate_2s_scores_interpolated(video_duration, model_scores)

    print(f"Video Duration: {video_duration}s")
    print(f"Segment Duration: {video_duration / len(model_scores):.2f}s per score")
    print("-" * 30)

    # The first 5 scores (covering time 0s to 10s)
    for i in range(5):
        start = i * 2
        end = start + 2
        print(f"Time [{start:0.1f}s - {end:0.1f}s) -> Score: {two_second_scores[i]:.4f}")

    # For [4.0, 6.0):
    # ((5.52-4.0)*0.8 + (6.0-5.52)*0.7)/2 = (1.52*0.8 + 0.48*0.7)/2 = (1.216 + 0.336)/2 = 0.776
    print("test_clip_score_generation OK")


def test_preds_to_spans_in_place():
    datadicts: list[dict[str, int | str]] = [
        {"preds": "1111111111111111111111111"},
        {"preds": "1111100111111111111111111"},
        {"preds": "1111100000101111000000000"},
        {"preds": "0000000000000000000000000"},
    ]
    for d in datadicts:
        d["duration"] = 150
    add_windows_to_masked_predictions(datadicts)
    p0 = datadicts[0]["pred_relevant_windows"] == [[0, 150, 1.0]]
    p0 = p0 and datadicts[1]["pred_relevant_windows"] == [[0, 30, 1.0], [42, 150, 1.0]]
    p0 = p0 and datadicts[2]["pred_relevant_windows"] == [[0, 30, 1.0], [60, 66, 1.0], [72, 96, 1.0]]
    p0 = p0 and datadicts[3]["pred_relevant_windows"] == []
    assert p0
    print("test_preds_to_spans_in_place OK")
    # print(datadicts)


def beam_search_to_scores(output, token_zero: int, token_one: int, num_frames: int):
    # get scores in format batch * num_beams, num_frames, vocab
    scores = torch.stack(output.scores).cpu().permute(1, 0, 2)[:, :num_frames, :]

    # get beam indices in shape: batch * num_beams, num_frames
    beam_indices = output.beam_indices.cpu()[:, :num_frames]

    # get beam scores in shape: batch, num_frames, vocab
    beam_scores = scores[beam_indices, torch.arange(scores.size(1)).unsqueeze(0)]

    # get probabilities of positive and negative
    beam_scores = beam_scores[:, :, [token_zero, token_one]].softmax(dim=2)

    # get positive probability
    score_ones = beam_scores[:, :, 1]
    return score_ones


def greedy_to_scores(output_scores: Tensor, token_zero: int, token_one: int, num_frames: int):
    # get scores in shape: batch, token, vocab
    scores = output_scores.cpu()

    # get scores in shape: batch, num_frames, 2
    scores = scores[:, :num_frames, [token_zero, token_one]]

    # get probabilities of positive and negative
    scores = scores.softmax(dim=2)

    # return positive probability
    return scores[:, :, 1]


# if __name__ == "__main__":
#     test_preds_to_spans_in_place()
#     test_clip_score_generation()


def generate_ranked_spans(data_dict: dict, num_proposal: int = 10) -> list:
    """
    Generates a ranked list of 10 spans by creating a large pool of candidate
    spans and scoring them, approximating a proposal-and-rank method.
    """
    scores = data_dict.get("score", [])
    duration = data_dict.get("duration", 1)
    if not scores:
        return [[0, 0, 0.0]] * num_proposal

    num_scores = len(scores)
    time_step = float(duration) / num_scores
    candidate_spans = set()

    # Spans from high-confidence thresholds
    for threshold in [0.9, 0.5, 0.1]:
        preds = "".join(["1" if s >= threshold else "0" for s in scores])
        in_span = False
        span_start_idx = -1
        for j, val in enumerate(preds):
            if val == "1" and not in_span:
                in_span = True
                span_start_idx = j
            elif val == "0" and in_span:
                in_span = False
                candidate_spans.add((span_start_idx, j))
        if in_span:
            candidate_spans.add((span_start_idx, num_scores))

    # Spans from random sliding windows of different scales
    # 2 & 4 = small, 8 & 12 = medium, 20 = long
    scales = [2, 4, 8, 12, 20]
    for scale in scales:
        if scale > num_scores:
            continue
        for i in range(num_scores - scale + 1):
            candidate_spans.add((i, i + scale))

    candidate_spans.add((0, num_scores))  # The entire video

    # Score Each Candidate Span
    scored_moments = []
    for start_idx, end_idx in candidate_spans:
        if start_idx >= end_idx:
            continue
        span_scores = scores[start_idx:end_idx]
        avg_score = float(sum(span_scores)) / len(span_scores)
        start_time = round(start_idx * time_step)
        end_time = round(end_idx * time_step)
        scored_moments.append([start_time, end_time, avg_score])

    # Rank, Select, and Pad
    sorted_moments = sorted(scored_moments, key=lambda x: x[2], reverse=True)
    if len(sorted_moments) > num_proposal:
        return sorted_moments[:num_proposal]
    else:
        num_to_pad = num_proposal - len(sorted_moments)
        dummy_moment = [0, 0, 0.0]
        return sorted_moments + ([dummy_moment] * num_to_pad)


def process_predictions_list_proposal(list_of_predictions: list[dict]) -> None:
    """Processes a list of prediction dictionaries in place."""
    for pred_dict in list_of_predictions:
        pred_dict["pred_relevant_windows"] = generate_ranked_spans(pred_dict)
        pred_dict["pred_saliency_scores"] = generate_2s_scores_interpolated(
            pred_dict.get("duration", 0), pred_dict.get("score", [])
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process prediction JSON file to generate moments and save as JSONL."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input predictions JSON file."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--method", type=str, choices=["proposal", "ranked"])
    parser.add_argument("--num_proposal", type=int, default=10)
    parser.add_argument("--video_path", type=str, help="to help insert video data incase missing")
    args = parser.parse_args()

    print(f"Input file: {args.input_path}")
    print(f"Output file: {args.output_path}")

    with open(args.input_path, "r") as f:
        predictions_data = json.load(f)
    print(f"Number of predictions: {len(predictions_data)}")

    if args.method == "ranked":
        process_predictions_list_sorted_moments(predictions_data, num_moments=args.num_proposal)
    else:
        process_predictions_list_proposal(predictions_data)

    if args.video_path is not None:
        with open(args.video_path, "r") as f:
            video_list = [json.loads(line) for line in f.readlines()]
        qid_to_vid = {v["qid"]: v["vid"] for v in video_list}
        for p in predictions_data:
            p["vid"] = qid_to_vid[p["qid"]]

    if args.clean:
        for pred in predictions_data:
            if "score" in pred:
                pred.pop("score")
            if "preds" in pred:
                pred.pop("preds")

    for p in predictions_data:
        assert len(p["pred_relevant_windows"]) == args.num_proposal

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w") as f:
        for item in predictions_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
