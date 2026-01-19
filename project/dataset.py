import copy
import json
import os
import random
from pathlib import Path
from typing import Literal, Sequence, TypedDict

import lightning as L
import numpy as np
import torch
import yaml
from torch.distributed import get_rank, get_world_size
from torch.multiprocessing import Value
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.io import ImageReadMode, decode_image
from transformers.tokenization_utils import PreTrainedTokenizer

from project.config import Config, DataConfig


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


def infer_tokens_from_tokenizer(tokenizer: PreTrainedTokenizer):
    token_0 = tokenizer.convert_tokens_to_ids("0")
    token_0 = token_0[-1] if isinstance(token_0, list) else token_0

    token_1 = tokenizer.convert_tokens_to_ids("1")
    token_1 = token_1[-1] if isinstance(token_1, list) else token_1

    token_assistant = tokenizer.convert_tokens_to_ids("<|assistant|>")
    token_assistant = token_assistant[-1] if isinstance(token_assistant, list) else token_assistant

    return token_0, token_1, token_assistant


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        with self.shared_epoch.get_lock():
            self.shared_epoch.value = epoch

    def get_value(self):
        with self.shared_epoch.get_lock():
            return self.shared_epoch.value


class VTGDataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig | Config,
        mode: Literal["fit", "validate", "test"],
        tokenizer: PreTrainedTokenizer,
        seed: int = 2109,
    ):
        self.do_train = mode == "fit"
        self.do_val = mode in ("fit", "validate")
        self.do_test = mode == "test"
        cfg_path = Path(cfg.base_data_dir) / cfg.yaml_path
        with open(cfg_path, "r") as f_in:
            # load with name as key
            data_cfg = yaml.safe_load(f_in)[cfg.dataset_name]

        self.tokenizer = tokenizer
        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.cfg = cfg

        self.train = MomentRetrievalDataset(cfg.base_data_dir, data_cfg, "train", cfg.num_train_samples)
        self.val = MomentRetrievalDataset(cfg.base_data_dir, data_cfg, "val", cfg.num_val_samples)
        self.test = MomentRetrievalDataset(cfg.base_data_dir, data_cfg, "test", cfg.num_test_samples)

        self.num_frames = self.train.num_frames

        world_size = get_world_size()
        distributed = world_size > 1

        self.shared_epoch = SharedEpoch()

        if distributed:
            self.val_sampler = DistributedSampler(
                self.val,
                num_replicas=world_size,
                rank=get_rank(),
                shuffle=False,
                drop_last=False,
                seed=seed,
            )

            self.train_sampler = DistributedSampler(
                self.train,
                num_replicas=world_size,
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
                seed=seed,
            )

        else:
            self.val_sampler = None
            self.train_sampler = None

        self.train_collator = TrainDataCollator(tokenizer)
        self.val_collator = InferenceDataCollator(tokenizer)
        self.test_collator = InferenceDataCollator(tokenizer)

        if self.do_train:
            self.train_collator([self.train[0], self.train[1]])
        if self.do_val:
            self.val_collator([self.val[0], self.val[1]])
        if self.do_test:
            self.test_collator([self.test[0], self.test[1]])

    def train_dataloader(self):
        assert self.do_train
        return DataLoader(
            dataset=self.train,
            batch_size=self.train_batch_size,
            num_workers=self.cfg.num_train_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            collate_fn=self.train_collator,
            persistent_workers=False,
            drop_last=True,
        )

    def val_dataloader(self):
        assert self.do_val
        return DataLoader(
            dataset=self.val,
            batch_size=self.val_batch_size,
            num_workers=self.cfg.num_val_workers,
            pin_memory=True,
            sampler=self.val_sampler,
            collate_fn=self.val_collator,
            persistent_workers=not self.do_train,  # training is always done on smaller machine with less CPU cores
            drop_last=False,
        )

    def test_dataloader(self):
        assert self.do_test
        return DataLoader(
            dataset=self.test,
            batch_size=self.val_batch_size,
            num_workers=self.cfg.num_val_workers,
            pin_memory=True,
            collate_fn=self.test_collator,
            persistent_workers=not self.do_train,
            drop_last=False,
        )
