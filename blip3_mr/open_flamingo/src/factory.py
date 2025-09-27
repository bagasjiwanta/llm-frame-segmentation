# copied from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/factory.py with some changes

import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from .utils import hasattr_recursive, setattr_recursive
from .xgenmm import XGenMMPerceiver

MODEL_ANYRES_GRIDS = [
    [384, 768],
    [768, 384],
    [768, 768],
    [1152, 384],
    [384, 1152],
]

"""
# Save the base model weights to use the local model
import os
import torch
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained(
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5", trust_remote_code=True
).vlm
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/xgenmm.pt")
"""

PRETRAINED_PATH = "weights/xgenmm.pt"

def load_pretrained(pretrained_path: str, model: nn.Module) -> dict:
    """
    Loads pretrained weights into a model from a checkpoint file.

    Args:
        pretrained (str): pretrained path
        model (nn.Module): The model instance to load the weights into.

    Returns:
        pretrained (dict): The state dictionary loaded from the checkpoint file.
    """
    assert isinstance(pretrained_path, str) and os.path.exists(pretrained_path)
    pretrained = torch.load(pretrained_path, map_location="cpu")

    if "vision_tokenizer.latents" in pretrained:
        msd_current = model.state_dict()
        if msd_current["vision_tokenizer.latents"].shape != pretrained["vision_tokenizer.latents"].shape:
            pretrained["vision_tokenizer.latents"] = msd_current["vision_tokenizer.latents"]  # Random re-init.

    if "vision_tokenizer.frame_embs" in pretrained:
        msd_current = model.state_dict()
        if msd_current["vision_tokenizer.frame_embs"] is None and pretrained["vision_tokenizer.frame_embs"] is not None:
            msd_current["vision_tokenizer.frame_embs"] = msd_current["vision_tokenizer.frame_embs"]

    result = model.load_state_dict(pretrained, strict=False)
    torch.cuda.empty_cache()
    rank = int(os.environ.get("RANK", 0))
    print(f"Rank {rank} Missing keys:", result.missing_keys)
    print(f"Rank {rank} Unexpected keys:", result.unexpected_keys)
    return pretrained


def create_model_and_tokenizer(
    vision_encoder_path: str = "google/siglip-so400m-patch14-384",
    lang_model_path: str = "microsoft/Phi-3-mini-4k-instruct",
    tokenizer_path: str = "microsoft/Phi-3-mini-4k-instruct",
    verbose: bool = True,
    num_vision_tokens: int = 128,
    image_aspect_ratio: str = "anyres",
    anyres_patch_sampling=True,
    gradient_checkpointing=True,
    pretrained: str | None = None 
) -> tuple[XGenMMPerceiver, PreTrainedTokenizer]:
    """
    Initialize XGenMMPerceiver model

    Args:
        vision_encoder_path (str): path to pretrained vision_encoder
        lang_model_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
        gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        verbose (bool, optional): whether to print model info. Defaults to True.
    Returns:
        `tuple[XGenMMPerceiver, PreTrainedTokenizer]`
    """

    # Configure dtypes
    vision_encoder_precision = torch.bfloat16
    lang_model_precision = torch.bfloat16

    attn_implementation = "flash_attention_2" if torch.cuda.get_device_capability(0)[0] >= 8 else "sdpa"

    vision_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=vision_encoder_path,
        torch_dtype=vision_encoder_precision,
        attn_implementation=attn_implementation,
    ).vision_model

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        legacy=False,
    )

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # add a pad token if it doesn't exist
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    lang_model = AutoModelForCausalLM.from_pretrained(
        lang_model_path,
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        torch_dtype=lang_model_precision,
    )

    check_embedding_fns(lang_model)

    # init the model
    decoder_layers_attr_name = "model.layers"

    model = XGenMMPerceiver(
        vision_encoder=vision_encoder,
        lang_model=lang_model,
        vis_feature_dim=vision_encoder.config.hidden_size,
        initial_tokenizer_len=len(tokenizer),
        decoder_layers_attr_name=decoder_layers_attr_name,
        pad_token_id=tokenizer.pad_token_id,
        anyres_grids=MODEL_ANYRES_GRIDS,
        anyres_patch_sampling=anyres_patch_sampling,
        gradient_checkpointing=gradient_checkpointing,
        num_vision_tokens=num_vision_tokens,
        image_aspect_ratio=image_aspect_ratio,
    )
    model.lang_model.to(lang_model_precision)

    # add special tokens to the tokenizer and language models
    tokenizer.add_special_tokens({"additional_special_tokens": list(model.special_tokens.values())})
    model.lang_model.config.vocab_size = len(tokenizer)
    model.set_special_token_ids({v: tokenizer.convert_tokens_to_ids(v) for v in model.special_tokens.values()})
    
    # freeze appropriate parameters
    model.set_trainable()

    # log model info
    if verbose and int(os.environ.get("RANK", 0)) == 0:
        print(f"BLIP-3 model initialized with {model.num_trainable_params:,} trainable parameters")
        print(f"==========Trainable Parameters\n{model.num_trainable_params_per_module}")
        print(f"==========Total Parameters\n{model.num_params_per_module}\n==========")

    load_pretrained(PRETRAINED_PATH if pretrained is None else pretrained, model)
    return model, tokenizer


def check_embedding_fns(lang_model):
    """Checks for and attempts to set {get/set}_{input/output}_embeddings functions to the model"""
    if not has_fn(lang_model, "get_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.get_input_embeddings = lambda: lang_model.transformer.wte
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.get_input_embeddings = lambda: lang_model.decoder.embed_tokens
        else:
            raise ValueError(
                "We require the language encoder to have a get_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(lang_model, "transformer.wte", x)
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(lang_model, "model.decoder.embed_tokens", x)
        else:
            raise ValueError(
                "We require the language encoder to have a set_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "get_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.get_output_embeddings = lambda: lang_model.lm_head
        else:
            raise ValueError(
                "We require the language encoder to have a get_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.set_output_embeddings = lambda x: setattr_recursive(lang_model, "lm_head", x)
        else:
            raise ValueError(
                "We require the language encoder to have a set_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )


def has_fn(model, fn_name):
    """Check if model has a function fn_name"""
    return callable(getattr(model, fn_name, None))
