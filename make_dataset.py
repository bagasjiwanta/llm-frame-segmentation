# turns a qvhighlights, charades style dataset into llava style.

"""Input folder structure (example for qvhighlights):
current_directory/
    datasets/  (dataset folder, based on console args)
        qvhighlights/  (dataset_name, based on console args)
            annotations/
                highlight_train_release.jsonl
                ...
            videos/
                vid1.mp4
                ...
        qvhighlights_v1/  (output folder, based on console args)
            p1/  (prompt style index + 1)
                test.json
                train.json
                val.json
            videos/
                vid1_frame001_var1.jpg
                vid1_frame001_var2.jpg
                ...
            video_summaries.json   (can be used to create another prompt version without parsing the videos all over again)
"""

# QVH style:
"""
{'qid': 10016,
 'query': 'Man in baseball cap eats before doing his interview.',
 'duration': 150,
 'vid': 'j7rJstUseKg_210.0_360.0',
 'relevant_clip_ids': [48, 49, 50, 51, 52, 53, 54, 55, 56],
 'saliency_scores': [[2, 3, 3],
  [4, 3, 2],
  [2, 3, 1],
  [2, 3, 0],
  [2, 3, 3],
  [2, 3, 2],
  [2, 3, 1],
  [2, 3, 0],
  [1, 3, 3]],
 'relevant_windows': [[96, 114]]}
"""

# Charades style:
"""
AO8RW 0.0 6.9##a person is putting a book on a shelf.
"""

# LLaVA style
"""
{
    "id": "000000033471",
    "image": "coco/train2017/000000033471.jpg",
    "conversations": [
        {
        "from": "user",
        "value": "Here are the frames: <image> <image> ... Which frames contains the activity: a person sitting near the train window?"
        },
        {
        "from": "assistant",
        "value": "010100100100111010111"
        },
    ]
}
"""
import argparse
import json
import multiprocessing
import os
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import List, Literal, TypedDict

import numpy as np
from decord import VideoReader, cpu, gpu
from PIL import Image
from tqdm import tqdm


def build_frames_newline(num_frames: int):
    return "\n".join(["<image>" for i in range(num_frames)])


def build_frames_zero_numbering(num_frames: int):
    return "\n".join([f"Frame {i}: <image>" for i in range(num_frames)])


def build_frames_spaces(num_frames: int):
    return " ".join(["<image>" for i in range(num_frames)])


PROMPT_TEMPLATE1 = """You are given {num_frames} frames sampled from a video, ordered and separated by newline characters:
{frames}

**Your task**: given an activity, analyze the video frames to identify which ones contain the specified activity.

**Output format**: provide a {num_frames} character binary segmentation mask, specifically:
 - '1' means the frame at that position likely matches to the activity.
 - '0' means the frame at that position likely does not match to the activity.
 - Your output must be exactly {num_frames} characters long and contain only '1's and '0's, with no spaces or other delimiters and no explanations.

**Activity**: {activity}

**Question**: Which frames contains the activity?
"""

PROMPT_TEMPLATE2 = """
You are given 25 frames sampled from a video, ordered and separated by newline characters, indexed from 0 to {last_frame_idx}:
{frames}

**Your task**: given an activity, analyze the video frames to identify which ones contain the specified activity.

**Output format**: provide a {num_frames} character binary segmentation mask, specifically:
 - '1' means the frame at that position likely matches to the activity.
 - '0' means the frame at that position likely does not match to the activity.
 - Your output must be exactly {num_frames} characters long and contain only '1's and '0's, with no spaces or other delimiters and no explanations.

**Activity**: {activity}

**Question**: Which frames contains the activity?
"""


class PromptStylesDict(TypedDict):
    instruction: str
    frame_builder: Callable


PROMPT_STYLES: list[PromptStylesDict] = [
    {"instruction": PROMPT_TEMPLATE1, "frame_builder": build_frames_newline},
    {"instruction": PROMPT_TEMPLATE2, "frame_builder": build_frames_zero_numbering},
]


NUM_CHARADES_TRAIN_SPLIT = 12408


@dataclass
class QVHighlightsData:
    qid: int
    query: str
    duration: int
    vid: str
    relevant_clip_ids: List[int]
    saliency_scores: List[List[int]]
    relevant_windows: List[List[int]]


@dataclass
class Args:
    prompt_style: int
    dataset_dir: str
    dataset_name: Literal["qvhighlights", "charades-sta"]
    num_frames: int
    num_workers: int
    processed_dir: str
    pretty_json: bool
    skip_video_processing: bool
    vid_in_dir: str
    ann_in_dir: str
    vid_out_dir: str
    ann_out_dir: str
    gpu: bool
    seed: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    frame_variation: int


def json_dumps(data, indent: int | None = 2, max_inline_length=120):
    json_str = json.dumps(data, indent=indent)
    # regex to find lists with no nested braces/brackets
    pattern = re.compile(r"\[\s*([^\[\]\{\}]+?)\s*\]", re.DOTALL)

    def replacer(match):
        content = match.group(1)
        # remove whitespace and newlines inside the list
        inline = " ".join(content.split())
        if len(inline) <= max_inline_length:
            return f"[ {inline} ]"
        else:
            return match.group(0)

    return pattern.sub(replacer, json_str)


def process_one_video_v(
    video_name: str,
    vid_in_dir: str,
    vid_out_dir: str,
    num_frames: int,
    frame_variation: int,
    use_gpu=True,
    single_file_name=True,
):
    filename = os.path.join(vid_in_dir, f"{video_name}.mp4")
    ctx = gpu(0) if use_gpu else cpu()
    with open(filename, "rb") as f_in:
        vr = VideoReader(f_in, ctx=ctx)

    fps = vr.get_avg_fps()
    total_frames = len(vr)

    step = total_frames / num_frames
    main_frame_indices = np.linspace(round(step / 2), total_frames - round(step / 2), num_frames, dtype=int)

    variation_sec = 0.25
    max_offset = variation_sec / 2
    offsets_sec = np.linspace(-max_offset, max_offset, frame_variation)
    offsets_frame = np.round(offsets_sec * fps).astype(int)  # offset in frame indices

    video_variants = []
    image_out_filenames = []

    for i, main_idx in enumerate(main_frame_indices):
        frame_indices = np.clip(main_idx + offsets_frame, 0, total_frames - 1)  # avoid OOB
        batch = vr.get_batch(frame_indices).asnumpy()

        for j, frame in enumerate(batch):
            out_filename = os.path.join(vid_out_dir, f"{video_name}_frame{i + 1:03d}_var{j + 1}.jpg")
            if not os.path.isfile(out_filename):
                Image.fromarray(frame).save(out_filename)
            if not single_file_name:
                image_out_filenames.append(os.path.basename(out_filename))

        video_variants.append(frame_indices)

    # Use timestamp of the main frame as reference
    video_times = vr.get_frame_timestamp(main_frame_indices).mean(-1).astype(int)

    if single_file_name:
        image_out_filenames = os.path.join(vid_out_dir, f"{video_name}")

    return (
        video_name,
        {
            "video_times": video_times,
            "image_out_filenames": image_out_filenames,
            "video_times_serial": video_times.tolist(),
        },
    )


def process_one_qvh(data: dict, num_frames: int, prompt_style: int, video_summaries: dict):
    video_summary = video_summaries[data["vid"]]
    video_times = video_summary["video_times"]
    if isinstance(video_times, list):
        video_times = np.array(video_times)
    is_test = "relevant_windows" not in data

    if not is_test:
        answer_times = np.array(data["relevant_windows"])
        binary_mask: np.ndarray | None = (
            np.any(
                (video_times[:, None] >= answer_times[:, 0]) & (video_times[:, None] <= answer_times[:, 1]),
                axis=1,
            )
            .astype(int)
            .tolist()
        )
    else:
        binary_mask = None

    # make the prompts
    activity = data["query"]

    prompt = PROMPT_STYLES[prompt_style]

    frames: str = prompt["frame_builder"](num_frames)

    user = prompt["instruction"].format(
        num_frames=num_frames, activity=activity, last_frame_idx=num_frames - 1, frames=frames
    )

    conversations = [{"role": "user", "content": user}]
    if not is_test and binary_mask is not None:
        assistant = "".join([str(b) for b in binary_mask])
        conversations.append({"role": "assistant", "content": assistant})

    output = {
        "id": data["qid"],
        "vid": data["vid"],
        "conversations": conversations,
        "video_timestamps": video_times.tolist(),
        "duration": data["duration"],
        "relevant_clip_ids": data['relevant_clip_ids']
    }

    if not is_test:
        output["relevant_windows"] = data["relevant_windows"]

    return output


def determine_split_sizes(args: Args):
    # Dataset specific logic
    splits: list[str] = []
    num_split_samples: list[int] = []
    if args.num_train_samples != 0:
        splits.append("train")
        num_split_samples.append(args.num_train_samples)
    if args.num_val_samples != 0 and args.dataset_name != "charades-sta":
        splits.append("val")
        num_split_samples.append(args.num_val_samples)
    if args.num_test_samples != 0:
        splits.append("test")
        num_split_samples.append(args.num_test_samples)
    return splits, num_split_samples


def process_dataset(args: Args):
    splits, num_split_samples = determine_split_sizes(args)
    print(f"\nWill process {', '.join(splits)} split(s) of {args.dataset_name} dataset")
    if args.dataset_name == "qvhighlights":
        src_file_format = "highlight_{split}_release.jsonl"
    else:  # args.dataset_name == 'charades-sta'
        src_file_format = "charades_sta_{split}.txt"

    input_paths = [os.path.join(args.ann_in_dir, src_file_format.format(split=split)) for split in splits]
    input_exists = all([os.path.isfile(path) for path in input_paths])
    assert input_exists, f"{args.dataset_name} dataset files are not found at \n{input_paths}"
    output_paths = [os.path.join(args.ann_out_dir, f"{split}.json") for split in splits]

    print(f"Will save the dataset split annotations to:")
    for o in output_paths:
        print(f" - {o}")

    split_data = []

    if args.dataset_name == "qvhighlights":
        for i in range(len(splits)):
            ann_in = input_paths[i]
            with open(ann_in, "r") as f_in:
                json_lines = [json.loads(line) for line in f_in.readlines()]
            num_split = num_split_samples[i]
            if num_split > 0 and num_split < len(json_lines):
                json_lines = random.choices(json_lines, k=num_split)
            split_data.append(json_lines)

    else:  # if charades-sta, then format to qvhighlights first
        for i in range(len(splits)):
            ann_in = input_paths[i]
            with open(ann_in, "r") as f_in:
                raw_lines = f_in.readlines()
            num_split = num_split_samples[i]
            if num_split != 0 and num_split < len(raw_lines):
                raw_lines = random.choices(raw_lines, k=num_split)
            json_lines = []
            for idx, raw_line in enumerate(raw_lines):
                line = raw_line.split("##")
                metadata = line[0]
                query = line[1].strip()
                metadatas = metadata.split(" ")
                st, ed = metadatas[1], metadatas[2]
                json_line = {
                    "vid": metadatas[0],
                    "qid": idx + NUM_CHARADES_TRAIN_SPLIT + 1 if i == 1 else idx,
                    "query": query,
                    "relevant_windows": [[st, ed]],
                }
                json_lines.append(json_line)

            split_data.append(json_lines)

    video_names: set = set()
    for split in split_data:
        video_names.update(set([r["vid"] for r in split]))

    video_summaries_dir = os.path.join(args.dataset_dir, args.processed_dir, "video_summaries.json")

    if not args.skip_video_processing:
        partial_process_one_video = partial(
            process_one_video_v,
            vid_in_dir=args.vid_in_dir,
            vid_out_dir=args.vid_out_dir,
            num_frames=args.num_frames,
            frame_variation=args.frame_variation,
            use_gpu=args.gpu,
        )
        video_summaries = {}
        video_summaries_serialized = {}  # video_times are np.ndarray
        with multiprocessing.Pool(args.num_workers) as pool:
            iterator = tqdm(
                pool.imap(partial_process_one_video, video_names),
                total=len(video_names),
                desc=f"Processing {len(video_names)} videos",
            )
            for vid, summary in iterator:
                video_summaries[vid] = {
                    "video_times": summary["video_times"],
                    "image_out_filenames": summary["image_out_filenames"],
                }
                video_summaries_serialized[vid] = {
                    "video_times": summary["video_times_serial"],
                    "image_out_filenames": summary["image_out_filenames"],
                }

        with open(video_summaries_dir, "w") as f_out:
            f_out.write(json_dumps(video_summaries_serialized, indent=2))
            print(f"\nSaved video summary to {video_summaries_dir}")

    else:  # if skip video processing
        with open(video_summaries_dir, "r") as f_in:
            print(f"Loading video summary from {video_summaries_dir}")
            video_summaries = json.loads(f_in.read())

    for i in range(len(splits)):
        raw_lines = split_data[i]
        processed = []
        worker_func = partial(
            process_one_qvh,
            num_frames=args.num_frames,
            prompt_style=args.prompt_style,
            video_summaries=video_summaries,
        )

        with multiprocessing.Pool(args.num_workers) as pool:
            for result in tqdm(
                pool.imap(worker_func, raw_lines), total=len(raw_lines), desc=f"Processing split {splits[i]}"
            ):
                processed.append(result)

        ann_out = output_paths[i]
        with open(ann_out, "w") as f_out:
            f_out.write(json_dumps(processed, indent=2 if args.pretty_json else None))

    print(f"\nDone. Outputs are saved to: \n{output_paths}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("--dataset_dir", required=True, default="datasets", type=str)
    parser.add_argument("--dataset_name", default="qvhighlights", choices=["qvhighlights", "charades-sta"])
    parser.add_argument("--num_frames", type=int, default=16, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pretty_json", default=True, action="store_true")
    parser.add_argument("--processed_dir", default="processed", type=str)
    parser.add_argument(
        "--skip_video_processing",
        default=False,
        action="store_true",
        help="If true, then use existing video_summaries-{processed_dir}.json (file must exist)",
    )
    parser.add_argument(
        "--gpu", default=False, help="Use gpu for decord video processing", action="store_true"
    )
    parser.add_argument(
        "--frame_variation", type=int, default=3, help="Add variations of images (+- 0.25 second)"
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=-1,
        help="Number of training samples to process. Negative number to process all available samples.",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=-1,
        help="Number of validation samples to process. Negative number to process all available samples.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=-1,
        help="Number of testing samples to process. Negative number to process all available samples.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args: Args = parser.parse_args()  # type: ignore
    return args


def main_process(args: Args):
    """Main process. Can be done with or without argparse"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.vid_in_dir = os.path.join(args.dataset_dir, args.dataset_name, "videos")
    args.ann_in_dir = os.path.join(args.dataset_dir, args.dataset_name, "annotations")
    args.vid_out_dir = os.path.join(args.dataset_dir, args.processed_dir, "videos")
    args.ann_out_dir = os.path.join(args.dataset_dir, args.processed_dir, f"p{args.prompt_style + 1}")
    print(f"Processing {args.dataset_name}\n")
    print(f"Args:")
    for k, v in vars(args).items():
        print(f" - {k}: {v}")

    if not os.path.isdir(args.vid_out_dir):
        os.makedirs(args.vid_out_dir)
    if not os.path.isdir(args.ann_out_dir):
        os.makedirs(args.ann_out_dir)

    process_dataset(args)


""" 
from make_dataset import Args, main_process   
config_without_argparse = Args(
    prompt_style=0,
    dataset_dir=".",
    dataset_name="qvhighlights",
    num_frames=25,
    num_workers=4,
    pretty_json=True,
    processed_dir="qvhighlights-25frames",
    skip_video_processing=False,
    gpu=False,
    seed=42,
    frame_variation=3,
    num_train_samples=0,
    num_val_samples=0,
    num_test_samples=-1,
    vid_in_dir=".",
    vid_out_dir=".",
    ann_in_dir=".",
    ann_out_dir="."
)
main_process(config_without_argparse)
"""


def main():
    args: Args = parse_arguments()
    main_process(args)


if __name__ == "__main__":
    main()
