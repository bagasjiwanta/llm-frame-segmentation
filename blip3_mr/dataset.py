import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as v2
import yaml
from torch.distributed import get_rank, get_world_size
from torch.multiprocessing import Value
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision.io import ImageReadMode, decode_image
from transformers.tokenization_utils import PreTrainedTokenizer

from blip3_mr.config import Config


class ConversationType(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RawDataType(TypedDict):
    id: int
    vid: str
    conversations: list[ConversationType]
    video_timestamps: list[int]
    duration: int
    relevant_windows: list[list[int]]
    saliency_scores: list[float]


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


SystemPrompt = (
    "<|system|>\n"
    "You are a smart video retrieval assistant. "
    "You will receive a video and a human activity query given by the user. "
    "Return the frames that matches the activity query. "
    "Follow the output format given by the user. "
    "<|end|>\n"
)

TrainingPromptTemplate = SystemPrompt + "<|user|>\n{prompt}<|end|>\n<|assistant|>\n{answer}<|end|>"

InferencePromptTemplate = SystemPrompt + "<|user|>\n{prompt}<|end|>\n<|assistant|>\n"


def make_img_resizer(device: str | int | torch.device):
    """image resizer that resizes the image to 384x384 (siglip's input size). Uses torchvision v2."""
    return v2.Resize((384, 384), interpolation=v2.InterpolationMode.BICUBIC).to(device)


def make_img_normalizer(
    compile_mode="default", device: str | int | torch.device = "cuda"
) -> torch._dynamo.OptimizedModule:
    """Image normalizer that normalizes to siglip's default value. (0.5 for mean and 0.5 for std)"""
    return torch.compile(
        torch.nn.Sequential(
            v2.ConvertImageDtype(dtype=torch.float32),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ),
        mode=compile_mode,
    ).to(device)


class MomentRetrievalDatasetOutput(TypedDict):
    """
    Output of a fast supervised dataset, containing prompts, ground truth answers,
    image data, and corresponding image sizes.

    Attributes:
        prompt (str): The text prompt associated with the data.
        answer (np.ndarray): The ground truth answer (segmentation mask)
        images (torch.Tensor): Image data in the shape of F, C, H, W
        image_sizes (list[tuple[int, int]]): A list of tuples, where each tuple contains
            the (width, height) of an image. The original xgenmm model expects (width, height)
        qid (int): The question id.
        relevant_windows (np.ndarray): The ground truth relevant windows in spans.
    """

    prompt: str
    answer: torch.Tensor | None
    images: torch.Tensor
    image_sizes: list[tuple[int, int]]
    qid: int
    relevant_windows: np.ndarray
    duration: int


class DatasetConfig(TypedDict):
    dir: str
    prompt: int
    variation: int
    num_frames: int
    use_variation: int
    version: int


class MomentRetrievalDataset(Dataset):
    def __init__(
        self,
        base_data_dir: str,
        dataset_config: DatasetConfig,
        split: Literal["train", "val", "test"] = "train",
        num_samples: int = 0,
        soft=True,
    ):
        super().__init__()
        self.dataset_path = os.path.join(base_data_dir, dataset_config["dir"])
        self.prompt = dataset_config.get("prompt", 1)
        self.variation = dataset_config.get("variation", 1)
        self.num_frames = dataset_config.get("num_frames", 16)
        self.use_variation = dataset_config.get("use_variation", 1)
        self.use_soft = soft
        self.split = split

        suffix = f"_v{dataset_config.get('version', 1)}"
        if split == "test":
            suffix = ""
        annotation_path = os.path.join(self.dataset_path, f"p{self.prompt}", f"{split}{suffix}.json")
        with open(annotation_path, "r") as f_in:
            annotations = json.load(f_in)

        if isinstance(annotations, dict):
            annotations = list(annotations.values())

        if num_samples > 0 and self.split == "train":
            num_samples = min(num_samples, len(annotations))
            annotations = random.sample(annotations, k=num_samples)

        if num_samples > 0 and self.split in ("val", "test"):
            num_samples = min(num_samples, len(annotations))
            annotations2 = []
            max_gap = len(annotations) // num_samples
            for i in range(0, num_samples * max_gap, max_gap):
                annotations2.append(annotations[i])
            annotations = annotations2

        # assert "saliency_scores" in annotations[0]
        self.images_path = os.path.join(self.dataset_path, "videos")
        assert os.path.isdir(self.images_path), (
            f"images path at {self.images_path} is not a directory. Perhaps you forgot to extract the videos ?"
        )
        self.list_data_dict: list[RawDataType] = annotations

    def __len__(self):
        return len(self.list_data_dict)

    def get_val_qvh(self) -> dict:
        """For offline validation with qvh API"""
        if self.split != "val":
            return {}

        dd = copy.deepcopy(self.list_data_dict)
        for i in range(len(dd)):
            dd[i]["label"] = dd[i]["conversations"][-1]["content"]
            dd[i].pop("conversations")
            dd[i]["qid"] = dd[i].pop("id")
            dd[i].pop("video_timestamps")
            if "_saliency_scores" in dd[i]:
                dd[i]["saliency_scores"] = dd[i]["_saliency_scores"]
                dd[i].pop("_saliency_scores")
        return {d["qid"]: d for d in dd}

    def __getitem__(self, i: int) -> MomentRetrievalDatasetOutput:
        data = copy.deepcopy(self.list_data_dict[i])

        image_file = data["vid"]
        if self.split == "train":
            image_idxs = np.random.randint(0, self.use_variation, self.num_frames)
        else:
            image_idxs = np.full(self.num_frames, self.use_variation // 2)
        image_idxs += np.arange(self.num_frames) * self.use_variation
        images = [
            f"{self.images_path}/{image_file}_frame{(i + 1):03d}_var{j + 1}.jpg"
            for i in range(self.num_frames)
            for j in range(self.variation)
        ]
        images = np.array(images)[image_idxs].tolist()

        image_list = [decode_image(image, mode=ImageReadMode.RGB) for image in images]
        image_sizes = [(image.size(2), image.size(1)) for image in image_list]
        image_tensor = torch.stack(image_list, dim=0)

        conversation: list = data["conversations"]
        prompt = conversation[0]["content"]

        if self.split == "train":
            prompt = TrainingPromptTemplate.format(prompt=prompt, answer=conversation[1]["content"])
        else:
            prompt = InferencePromptTemplate.format(prompt=prompt)

        answer = None
        if len(conversation) > 1:  # train and val
            if self.use_soft:
                answer = torch.stack(
                    [
                        1.0 - torch.tensor(data["saliency_scores"]),
                        torch.tensor(data["saliency_scores"]),
                    ],
                    dim=1,
                )
            else:
                answer = torch.tensor([int(a) for a in conversation[1]["content"]])

        if "relevant_windows" in data:
            relevant_windows = np.array(data["relevant_windows"])
        else:
            relevant_windows = None

        return {
            "prompt": prompt,
            "answer": answer,
            "images": image_tensor,
            "image_sizes": image_sizes,
            "qid": int(data["id"]),
            "relevant_windows": relevant_windows,
            "duration": int(data["duration"]),
        }


@dataclass
class DataInfo:
    """
    DataInfo is a dataclass that holds information about a dataset.
    """

    name: str
    dataloader: DataLoader
    dataset: MomentRetrievalDataset
    tokenizer: PreTrainedTokenizer
    sampler: torch.utils.data.Sampler | None = None
    shared_epoch: SharedEpoch | None = None
    num_frames: int = 0
    token_zero: int = 29900
    token_one: int = 29896
    token_assistant: int = 32001

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class TrainCollatorOutput(TypedDict):
    """
    Output of the training data collator

    Attributes:
        pixel_values (list[torch.Tensor]): videos in shape of [B], F, C, H, W
        input_ids (torch.Tensor): language input ids in the shape of B, T
        attention_mask (torch.Tensor): attention masks in the shape of B, T
        image_size (list[list[list[int]]]): image sizes
        labels (torch.Tensor): language labels in the shape of B, T
        answers (np.ndarray): ground truth segmentation mask
    """

    pixel_values: list[torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_size: list[list[tuple[int, int]]]
    labels: torch.Tensor
    answers: torch.Tensor


class TrainDataCollator:
    """
    Data collator for training.

    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for the data collator.
        assistant_token (int): The token id of the assistant token.
        pad_token (int): The token id of the pad token.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.assistant_token = self.tokenizer.added_tokens_encoder["<|assistant|>"]
        self.pad_token = self.tokenizer.added_tokens_encoder["<pad>"]

    def __call__(self, raw_batch: Sequence[MomentRetrievalDatasetOutput]) -> TrainCollatorOutput:
        image_sizes = [item["image_sizes"] for item in raw_batch]
        prompts = [item["prompt"] for item in raw_batch]
        answers = torch.stack([item["answer"] for item in raw_batch], dim=0)

        # Phi-3 is trained using padding_side='left', avoiding distribution shift
        # Pad to multiple of 8 is used to utilize tensor cores
        lang_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding_side="left", padding=True, pad_to_multiple_of=8
        )
        input_ids: torch.Tensor = lang_inputs["input_ids"]  # type: ignore

        _, T = input_ids.shape
        # grabs the first position of <assistant> token in each batch
        assistant_pos = (input_ids == self.assistant_token).float().argmax(dim=1)
        arange = torch.arange(T, device=input_ids.device).unsqueeze(0)

        # needed to be >= since logits predict the next token
        ignore_mask = arange >= assistant_pos.unsqueeze(1)
        pad_mask = input_ids != self.pad_token
        mask = ignore_mask & pad_mask

        labels = input_ids.clone()
        labels[~mask] = -100

        images = [item["images"] for item in raw_batch]

        return {
            "pixel_values": images,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": lang_inputs["attention_mask"],  # type: ignore
            "image_size": image_sizes,
            "answers": answers,
        }


class InferenceCollatorOutput(TypedDict):
    """
    Output of the inference data collator

    Attributes:
        pixel_values (torch.Tensor): videos in shape of [B], F, C, H, W
        input_ids (torch.Tensor): language input ids in the shape of B, T
        attention_mask (torch.Tensor): attention masks in the shape of B, T
        image_size (list[list[tuple[int, int]]]): image sizes
        labels (np.ndarray): ground truth segmentation mask (assuming this is equivalent to 'answers' in training)
    """

    pixel_values: list[torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_size: list[list[tuple[int, int]]]
    labels: torch.Tensor | None
    qids: np.ndarray
    relevant_windows: list[np.ndarray]
    durations: np.ndarray


class InferenceDataCollator:
    """
    InferenceDataCollator is a data collator for inference (validation and test).
    See InferenceCollatorOutput for the __call__ method's output type.

    Attributes:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for the data collator.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, raw_batch) -> InferenceCollatorOutput:
        prompts = [item["prompt"] for item in raw_batch]
        if not any([item["answer"] is None for item in raw_batch]):
            labels = torch.stack([item["answer"] for item in raw_batch], dim=0)
        else:
            labels = None
        image_sizes = [item["image_sizes"] for item in raw_batch]
        qids = np.array([item["qid"] for item in raw_batch])
        relevant_windows = [item["relevant_windows"] for item in raw_batch]
        durations = np.array([item["duration"] for item in raw_batch])

        # Phi-3 is trained using padding_side='left', avoiding distribution shift
        # Pad to multiple of 8 is used to utilize tensor cores
        lang_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding_side="left", padding=True, pad_to_multiple_of=8
        )
        input_ids: torch.Tensor = lang_inputs["input_ids"]  # type: ignore
        attention_mask: torch.Tensor = lang_inputs["attention_mask"]  # type: ignore

        images = [item["images"] for item in raw_batch]
        return {
            "pixel_values": images,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "image_size": image_sizes,
            "qids": qids,
            "relevant_windows": relevant_windows,
            "durations": durations,
        }


class StratifiedBatchSampler(Sampler):
    """
    Stratifies batches based only on the 'non_zero' feature.
    """

    def __init__(
        self,
        dataset: MomentRetrievalDataset,
        batch_size: int,
        n_bins: int = 4,
        shuffle: bool = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset.list_data_dict
        self.batch_size = batch_size
        self.shuffle = shuffle

        # list of "001100101111"
        non_zeros = [d["conversations"][-1]["content"] for d in self.dataset]
        # list of [False, False, True, ...]
        non_zeros = [[s == "1" for s in d] for d in non_zeros]
        # sum items and convert into array
        non_zeros = np.array([sum(d) for d in non_zeros])

        non_zero_bins = np.digitize(non_zeros, bins=np.linspace(non_zeros.min(), non_zeros.max(), n_bins + 1)) - 1
        non_zero_bins[non_zero_bins == n_bins] = n_bins - 1

        self.stratify_groups = non_zero_bins
        self.num_groups = self.stratify_groups.max() + 1

        self.grouped_indices = [[] for _ in range(self.num_groups)]
        for i, group in enumerate(self.stratify_groups):
            self.grouped_indices[group].append(i)

    def __iter__(self):
        indices_by_group = [list(group) for group in self.grouped_indices]

        if self.shuffle:
            for group in indices_by_group:
                random.shuffle(group)

        # Create a single pool of indices by interleaving the groups
        group_pool = []
        group_iters = [iter(group) for group in indices_by_group]
        while True:
            active_iters = 0
            for it in group_iters:
                try:
                    group_pool.append(next(it))
                    active_iters += 1
                except StopIteration:
                    pass
            if active_iters == 0:
                break

        all_batches = [group_pool[i : i + self.batch_size] for i in range(0, len(group_pool), self.batch_size)]
        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DistributedStratifiedBatchSampler(DistributedSampler):
    """
    A Distributed Sampler that stratifies batches based only on the 'non_zero' feature.

    Yields individual batches like `torch.utils.data.BatchSampler`
    """

    def __init__(
        self,
        dataset: MomentRetrievalDataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        batch_size: int = 8,
        n_bins=4,
        seed=42,
        drop_last=True,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires init_distributed to be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Requires init_distributed package to be initialized")
            rank = dist.get_rank()

        self.dataset = dataset.list_data_dict
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

        # list of "001100101111"
        non_zeros = [d["conversations"][-1]["content"] for d in self.dataset]
        # list of [False, False, True, ...]
        non_zeros = [[s == "1" for s in d] for d in non_zeros]
        # sum items and convert into array
        non_zeros = np.array([sum(d) for d in non_zeros])

        non_zero_bins = np.digitize(non_zeros, bins=np.linspace(non_zeros.min(), non_zeros.max(), n_bins + 1)) - 1
        non_zero_bins[non_zero_bins == n_bins] = n_bins - 1

        self.stratify_groups = non_zero_bins
        self.num_groups = self.stratify_groups.max() + 1

        self.grouped_indices = [[] for _ in range(self.num_groups)]
        for i, group in enumerate(self.stratify_groups):
            self.grouped_indices[group].append(i)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices_by_group = [list(group) for group in self.grouped_indices]

        if self.shuffle:
            for i, group in enumerate(indices_by_group):
                group_perm = torch.randperm(len(group), generator=g).tolist()
                indices_by_group[i] = [group[j] for j in group_perm]

        # Create a single, stratified pool of indices
        global_indices_pool = []
        group_iters = [iter(group) for group in indices_by_group]
        while True:
            active_iters = 0
            for it in group_iters:
                try:
                    global_indices_pool.append(next(it))
                    active_iters += 1
                except StopIteration:
                    pass
            if active_iters == 0:
                break

        if self.drop_last:
            usable_size = len(global_indices_pool) - (len(global_indices_pool) % self.batch_size)
        else:
            usable_size = len(global_indices_pool)

        all_batches = [
            global_indices_pool[i : i + self.batch_size]
            for i in range(0, usable_size, self.batch_size)
        ]

        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        if not self.drop_last:
            total_size = math.ceil(len(all_batches) / self.num_replicas) * self.num_replicas
            padding_size = total_size - len(all_batches)
            if padding_size > 0:
                all_batches += all_batches[:padding_size]

        batches_for_this_replica = all_batches[self.rank : len(all_batches) : self.num_replicas]

        return iter(batches_for_this_replica)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // (self.batch_size * self.num_replicas)
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_replicas))

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def infer_tokens_from_tokenizer(tokenizer: PreTrainedTokenizer):
    token_zero = tokenizer.convert_tokens_to_ids("0")
    token_zero = token_zero[-1] if isinstance(token_zero, list) else token_zero

    token_one = tokenizer.convert_tokens_to_ids("1")
    token_one = token_one[-1] if isinstance(token_one, list) else token_one

    token_assistant = tokenizer.convert_tokens_to_ids("<|assistant|>")
    token_assistant = token_assistant[-1] if isinstance(token_assistant, list) else token_assistant

    return token_zero, token_one, token_assistant


def make_test_datainfo(tokenizer: PreTrainedTokenizer, config: Config, verbose=True):
    """Makes a test datainfo. Expected to be ran in 1 GPU only."""
    config_path = os.path.join(config.base_data_dir, config.dataset_config)
    dataset_name = config.test_dataset_name if config.test_dataset_name is not None else config.dataset_name
    with open(config_path, "r") as f_in:
        dataset_config: DatasetConfig = yaml.safe_load(f_in)[dataset_name]

    test_dataset = MomentRetrievalDataset(
        config.base_data_dir, dataset_config, "test", num_samples=config.num_test_samples
    )
    _test_1, _test_2 = test_dataset[0], test_dataset[1]

    test_collator = InferenceDataCollator(tokenizer)
    _test = test_collator([_test_1, _test_2])

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.val_batch_size,
        num_workers=config.num_val_workers,
        pin_memory=True,
        collate_fn=test_collator,
        persistent_workers=True,
        drop_last=False,
        shuffle=False,
    )
    tokens = infer_tokens_from_tokenizer(tokenizer)

    test_datainfo = DataInfo(
        name=config.dataset_name + "_test",
        dataloader=test_dataloader,
        num_frames=dataset_config["num_frames"],
        token_zero=tokens[0],
        token_one=tokens[1],
        token_assistant=tokens[2],
        tokenizer=tokenizer,
        dataset=test_dataset,
    )

    if config.rank == 0 and verbose:
        print("Test collator input_ids, attention_mask, and prompt (first item):")
        print(_test["input_ids"][0, :])
        print(_test["attention_mask"][0, :], end="\n\n")
        print(tokenizer.batch_decode(_test["input_ids"], skip_special_tokens=True)[0], end="\n\n")
        print(f"Num test iter: {len(test_dataloader)}")
        print(f"Num test samples: {len(test_dataset)}")

    return test_datainfo


def make_train_val_datainfos(
    tokenizer: PreTrainedTokenizer, config: Config, distributed=True, verbose=True
) -> tuple[DataInfo, DataInfo]:
    """
    Creates train and val datainfos for the given dataset.

    Args:
        tokenizer (LlamaTokenizer): The tokenizer to use for the dataset.
        config (Config): The config to use for the dataset.
        distributed (bool): Whether to use distributed training.
    """
    if config.world_size > 1 and not distributed and dist.is_initialized():
        distributed = True

    # Load the dataset config
    config_path = os.path.join(config.base_data_dir, config.dataset_config)
    with open(config_path, "r") as f_in:
        dataset_config = yaml.safe_load(f_in)[config.dataset_name]

    # --- Build and test datasets ---
    train_dataset = MomentRetrievalDataset(config.base_data_dir, dataset_config, "train", config.num_train_samples)
    val_dataset = MomentRetrievalDataset(config.base_data_dir, dataset_config, "val", config.num_val_samples)

    _test_train0 = train_dataset[0]
    _test_train1 = train_dataset[1]
    _test_val0 = val_dataset[0]
    _test_val1 = val_dataset[1]

    # --- Build sampler ---
    train_batch_sampler = None
    train_sampler = None
    val_sampler = None

    if distributed:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
            seed=config.seed,
        )

        if config.sampler == "pytorch" or config.sampler is None:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
                seed=config.seed,
            )

        elif config.sampler == "stratified":
            train_batch_sampler = DistributedStratifiedBatchSampler(
                train_dataset,
                config.world_size,
                config.rank,
                shuffle=True,
                batch_size=config.train_micro_batch_size_per_gpu,
                n_bins=5,
                seed=config.seed,
            )
        else:
            raise NotImplementedError(f"sampler {config.sampler} is not valid")

    else:  # not distributed
        if config.sampler == "stratified":
            train_batch_sampler = StratifiedBatchSampler(
                dataset=train_dataset, batch_size=config.train_micro_batch_size_per_gpu, n_bins=5
            )

    train_micro_batch_size = config.train_micro_batch_size_per_gpu
    drop_last = True
    if train_batch_sampler is not None:
        train_sampler = None
        train_micro_batch_size = 1
        drop_last = False

    # --- Build and test collators ---
    train_collator = TrainDataCollator(tokenizer)
    val_collator = InferenceDataCollator(tokenizer)

    _test_train = train_collator([_test_train0, _test_train1])
    if config.rank == 0 and verbose and config.do_train:
        print("Train collator labels, input_ids, and attention_mask (first item):")
        print("labels.shape =", _test_train["labels"].shape)
        print(_test_train["labels"][_test_train["labels"] != -100].view(2, -1)[0])
        print(_test_train["input_ids"][0, :])
        print(_test_train["attention_mask"][0, :])
    _test_val = val_collator([_test_val0, _test_val1])
    if config.rank == 0 and verbose and config.do_val:
        print("Val collator labels, input_ids, and attention_mask (first item):")
        print("labels.shape =", _test_val["labels"].shape)
        print(_test_val["labels"][0])
        print(_test_val["input_ids"][0, :])
        print(_test_val["attention_mask"][0, :])

    num_train_frames = _test_train["pixel_values"][0].size(0)
    num_val_frames = _test_val["pixel_values"][0].size(0)
    assert num_train_frames == num_val_frames

    # --- DataLoaders ---
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_micro_batch_size,
        num_workers=config.num_train_workers,
        pin_memory=True,
        sampler=train_sampler,
        batch_sampler=train_batch_sampler,
        collate_fn=train_collator,
        persistent_workers=False,
        drop_last=drop_last,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val_batch_size,
        num_workers=config.num_val_workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=val_collator,
        persistent_workers=not config.do_train,  # use persistent worker if only validation
        drop_last=False,
    )

    if config.rank == 0:
        print(f"Num train iter per epoch: {len(train_dataloader)}. Num val iter per epoch: {len(val_dataloader)}")

    token_zero, token_one, token_assistant = infer_tokens_from_tokenizer(tokenizer)

    train_datainfo = DataInfo(
        name=config.dataset_name + "_train",
        dataloader=train_dataloader,
        sampler=train_sampler,
        shared_epoch=None,
        num_frames=num_train_frames,
        token_zero=token_zero,
        token_one=token_one,
        token_assistant=token_assistant,
        tokenizer=tokenizer,
        dataset=train_dataset,
    )
    val_datainfo = DataInfo(
        name=config.dataset_name + "_val",
        dataloader=val_dataloader,
        sampler=val_sampler,
        shared_epoch=None,
        num_frames=num_val_frames,
        token_zero=token_zero,
        token_one=token_one,
        token_assistant=token_assistant,
        tokenizer=tokenizer,
        dataset=val_dataset,
    )

    return train_datainfo, val_datainfo


def train_batch_to_device(
    batch: TrainCollatorOutput, device: torch.device | str | int
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Moves the batch to the given device_id.

    Args:
        batch (FastTrainCollatorOutput): The batch to move to the given device.
        device_id (torch.device | str | int): The device to move the batch to.

    Returns:
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The moved batch.
    """

    # --- Images are expected to be [B], F, C, H, W
    images = [image.to(device, non_blocking=True) for image in batch["pixel_values"]]

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    answers = batch["answers"].to(device, non_blocking=True)  # Batch, Frame

    return (images, input_ids, attention_mask, labels, answers)


def val_batch_to_device(
    batch: InferenceCollatorOutput, device: torch.device | str | int
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Moves the batch to the given device_id.

    Args:
        batch (FastInferenceCollatorOutput): The batch to move to the given device.
        device_id (torch.device | str | int): The device to move the batch to.

    Returns:
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]: The moved batch.
    """
    # --- Images are expected to be [B], F, C, H, W
    images = [image.to(device, non_blocking=True) for image in batch["pixel_values"]]

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    return (images, input_ids, attention_mask)


def process_images(images: list[torch.Tensor], img_resizer: torch.nn.Module, img_normalizer: torch.nn.Module):
    # resize each image so that it becomes 384, 384, the dimension is now [Batch], Frame, Channel, Height, Width
    image_list = [img_resizer(frames) for frames in images]
    # stack to remove the list
    image_stack = torch.stack(image_list, dim=0)
    # normalize the image in a single process
    image_stack = img_normalizer(image_stack)
    # create 2 new dimension, shape is Batch, Frame, 1, 1, Channel, Height, Width
    image_stack = image_stack.unsqueeze(2).unsqueeze(2)
    # unbind to convert first two shape to list, final shape is [Batch], [Frame], 1, 1, Channel, Height, Width (xgen-mm expects these 2 extra dims)
    image_double_list = [list(torch.unbind(image, dim=0)) for image in image_stack]
    return image_double_list
