import argparse
import json
import os
from typing import TypedDict

import torch
from torch import Tensor

# from transformers.generation.utils import BeamSearchDecoderOnlyOutput


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(_l.strip("\n")) for _l in f.readlines()]


class pred_datadict(TypedDict):
    preds: str
    duration: int
    score: list[float]
    pred_relevant_windows: list[list[int]] | None
    pred_saliency_scores: list[int] | None


def inplace_pred_string_to_relevant_windows(data_dict: list[pred_datadict], num_moments: int = 10) -> None:
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

        spans = sorted(spans, key=lambda span: span[-1], reverse=True)
        remaining = num_moments - len(spans)
        if remaining > 0:
            padding = [[0, 150, 0.0] for _r in range(remaining)]
            spans.extend(padding)

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
    inplace_pred_string_to_relevant_windows(datadicts)
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


def main():
    parser = argparse.ArgumentParser(description="Process prediction JSON file to generate moments and save as JSONL.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input predictions JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--num_proposal", type=int, default=10)
    parser.add_argument("--video_path", type=str, help="to help insert video data incase missing")
    args = parser.parse_args()

    print(f"Input file: {args.input_path}")
    print(f"Output file: {args.output_path}")

    with open(args.input_path, "r") as f:
        predictions_data = json.load(f)
    print(f"Number of predictions: {len(predictions_data)}")

    inplace_pred_string_to_relevant_windows(predictions_data, num_moments=args.num_proposal)

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
        for index, item in enumerate(predictions_data):
            if index == len(predictions_data) - 1:
                f.write(json.dumps(item))
            else:
                f.write(json.dumps(item) + "\n")

    print(f"Saved results to: {args.output_path}")


if __name__ == "__main__":
    main()
