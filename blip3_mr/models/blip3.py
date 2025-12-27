# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeGuard, TypeVar, Union, cast

# from blip3_mr.open_flamingo.train.any_res_data_utils import get_anyres_image_grid_shape, unpad_image
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many
from PIL import Image
from torch import einsum, nn
from transformers import CLIPVisionModel, Phi3ForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer


def unpad_image(tensor, original_size, keep_original_shape=False):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        if keep_original_shape:
            attention_mask = torch.ones((current_height, current_width), device=tensor.device)
            attention_mask[:padding, :] = 0
            attention_mask[current_height - padding :, :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
            return unpadded_tensor, None
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        if keep_original_shape:
            attention_mask = torch.ones((current_height, current_width), device=tensor.device)
            attention_mask[:, :padding] = 0
            attention_mask[:, current_width - padding :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, :, padding : current_width - padding]
            return unpadded_tensor, None


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # FIXME: determine grid_pinpoints from image sizes.
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    processor_size = processor.transforms[0].size
    patches = divide_to_patches(image_padded, processor_size[0])

    image_original_resize = image.resize((processor_size[0], processor_size[0]))

    image_patches = [image_original_resize] + patches
    image_patches = [processor(image_patch) for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.transforms[-1].mean))
            image = image_processor(image)
            new_images.append(image)
    elif image_aspect_ratio in ["anyres", "anyres-legacy"]:
        base_img_size = image_processor.transforms[0].size[0]
        for image in images:
            image = process_anyres_image(
                image,
                image_processor,
                [
                    [base_img_size, base_img_size * 2],
                    [base_img_size * 2, base_img_size],
                    [base_img_size * 2, base_img_size * 2],
                    [base_img_size * 3, base_img_size],
                    [base_img_size, base_img_size * 3],
                ],
            )

            new_images.append(image)
    else:
        return image_processor(images)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


# copied directly from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/helpers.py

"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""


@dataclass
class VLMOutputWithPast(CausalLMOutputWithPast):
    """
    VLMOutputWithPast is a wrapper around CausalLMOutputWithPast that adds the following attributes:
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
    """

    past_media_locations: Optional[torch.Tensor] = None
    past_vision_tokens: Optional[torch.Tensor] = None


T = TypeVar("T")


def exists(val: T | None) -> TypeGuard[T]:
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)  # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat(
                (
                    vision_attn_masks,
                    torch.ones((latents.shape[0], latents.shape[-2]), dtype=latents.dtype, device=latents.device),
                ),
                dim=-1,
            )
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros((q.size(0), 1, 1, q.size(-2), k.size(-2)), dtype=q.dtype, device=q.device)
            vision_attn_masks = repeat(vision_attn_masks, "b n -> b 1 1 l n", l=q.size(-2))
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=128,
        max_num_media: int | None = None,
        max_num_frames: int | None = None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, vision_attn_masks):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents

        if exists(self.projection):
            return self.projection(self.norm(latents))
        else:
            return self.norm(latents)


class LinearPatchProjection(VisionTokenizer):
    """Linear projection from patch features to image tokens."""

    def __init__(self, mm_projector_type, *, dim_visual, dim_out, num_patches):
        super().__init__(dim_media=dim_visual, num_tokens_per_media=num_patches)
        if mm_projector_type == "linear":
            self.proj = nn.Linear(dim_visual, dim_out)
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(dim_visual, dim_out)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(dim_out, dim_out))
                self.proj = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x):
        B = x.shape[0]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")
        x = self.proj(x)
        return rearrange(x, "(b T) n d -> b T n d", b=B)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt_all)
                T_txt_all >= T_txt
                If T_txt_all > T_txt, then the last T_txt text_times are used
        """

        T_txt = x.shape[1]
        assert T_txt <= media_locations.shape[1], "current text cannot be longer than conditioned media locations"

        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            media_time = torch.arange(T_img, device=x.device) + 1

            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)[:, -T_txt:]

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x


class QFormerWithProjection(VisionTokenizer):
    """
    Based on BLIP-2 (https://arxiv.org/pdf/2301.12597.pdf)
    In the BLIP-2 paper, Q-former is initialized with BERT-base weights,
    so dim_inner = 768, num_hidden_layers = 12, and intermediate_size = 3072
    """

    def __init__(
        self,
        dim_input,
        dim_out,
        dim_inner=768,
        num_hidden_layers=12,
        num_query_tokens=32,
    ):
        super().__init__(dim_media=dim_out, num_tokens_per_media=num_query_tokens)
        # initialize the qformer
        from transformers import Blip2QFormerConfig, Blip2QFormerModel

        self.qformer = Blip2QFormerModel(
            Blip2QFormerConfig(
                encoder_hidden_size=dim_input,
                hidden_size=dim_inner,
                num_hidden_layers=num_hidden_layers,
            )
        )
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, dim_inner))
        self.proj = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (B, T, F, v, D)
        Returns:
            shape (B, T, n, D) where n is num_query_tokens
        """
        # HF class expects three dimensional input
        B, T = x.shape[:2]
        x = rearrange(x, "b T F v d -> (b T) (F v) d")

        # get the outputs
        image_attention_mask = torch.ones(x.size()[:-1], dtype=torch.long, device=x.device)
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        query_output = query_outputs[0]
        query_output = self.proj(query_output)

        # reshape
        query_output = rearrange(query_output, "(b T) n d -> b T n d", b=B)
        return query_output


# Both DecoupledEmbedding and DecoupledLinear are taken from https://github.com/huggingface/transformers/blob/v4.32.1/src/transformers/models/idefics/modeling_idefics.py and renamed for clarity
class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        max_original_id: int,
        num_additional_embeddings: int = 0,
        _weight: torch.Tensor = None,
        num_original_embeddings: int = None,
        embedding_dim: int = None,
        partially_freeze=True,
        device=None,
        dtype=None,
        pad_token_id=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`):
                The largest token id that should be embedded using the regular embedding (regular `weight`).
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal self.weight.shape[0]
            num_additional_embeddings (`int`):
                Number of additional tokens to initialize an Embedding matrix for (`additional_weight`).
            _weight (`torch.Tensor`, *optional*, defaults to `None`): The regular weight tensor.
                If provided, this sets the `num_original_embeddings` and `embedding_dim` parameters.
            num_original_embeddings (`int`):
                self.weight.shape[0]
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `True`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # validate args
        if pad_token_id is not None and pad_token_id > max_original_id:
            raise ValueError(
                f"pad_token_id must be <= max_original_id. Got {pad_token_id} and {max_original_id}."
                + "If the original tokenizer does not have a pad_token_id, use pad_token_id=None."
            )
        if _weight is not None:
            assert (num_original_embeddings is None) or (_weight.shape[0] == num_original_embeddings), (
                f"num_original_embeddings={num_original_embeddings} but _weight.shape[0]={_weight.shape[0]}"
            )
            assert (embedding_dim is None) or (_weight.shape[1] == embedding_dim), (
                f"embedding_dim={embedding_dim} but _weight.shape[1]={_weight.shape[1]}"
            )
            num_original_embeddings = _weight.shape[0]
            embedding_dim = _weight.shape[1]
        else:
            assert num_original_embeddings is not None, (
                "num_original_embeddings must be provided if _weight is not provided"
            )
            assert embedding_dim is not None, "embedding_dim must be provided if _weight is not provided"

        super().__init__(
            num_embeddings=num_original_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=pad_token_id,
            _weight=_weight,
        )
        self.max_original_id = max_original_id
        self.padding_idx = pad_token_id
        self.num_additional_embeddings = num_additional_embeddings
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(require_regular_grad=not partially_freeze, require_additional_grad=True)

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        self.additional_embedding.requires_grad_(require_additional_grad)

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
        embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids > self.max_original_id)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.max_original_id - 1)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_original_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.max_original_id + 1,
            self.num_additional_embeddings,
            self.embedding_dim,
            (not self.weight.requires_grad),
        )


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `additional_out_features` > 0,
    then it will create `additional_out_features * in_features` additional parameters that are always trained. If
    `additional_out_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        max_original_id: int,
        additional_out_features: int = 0,
        _weight: torch.Tensor = None,
        _bias: torch.Tensor = None,
        in_features: int = None,
        original_out_features: int = None,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`): The largest token id that should be extracted from the regular weight.
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal original_out_features - 1
            _weight: torch.Tensor, *optional*, defaults to `None`. The regular weight tensor.
                If provided, this sets the `in_features` and `original_out_features` parameters.
            _bias: torch.Tensor, *optional*, defaults to `None`. The regular bias tensor.
            in_features: int. Input hidden size.
            original_out_features: int. Original out_features of the language model's get_output_embeddings() function.
            additional_out_features: int. Number of additional trainable dimensions.
            bias: bool. Whether to include a bias term.
            partially_freeze: bool, *optional*, defaults to `True`): If `True`, the regular `weight` will be frozen.
        """
        # argument validation
        if _weight is not None:
            assert (_weight.shape[0] == original_out_features) or (original_out_features is None), (
                f"original_out_features={original_out_features} but _weight.shape[0]={_weight.shape[0]}"
            )
            assert (_weight.shape[1] == in_features) or (in_features is None), (
                f"in_features={in_features} but _weight.shape[1]={_weight.shape[1]}"
            )
            in_features = _weight.shape[1]
            original_out_features = _weight.shape[0]
        else:
            assert in_features is not None, "in_features must be provided if _weight is not provided"
            assert original_out_features is not None, (
                "original_out_features must be provided if _weight is not provided"
            )

        if _bias is not None:
            assert bias is True, "bias must be True if _bias is provided"

        # initialize original linear
        super().__init__(in_features, original_out_features, bias, device, dtype)

        # set weight and bias manually
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        if _bias is not None:
            self.bias = nn.Parameter(_bias)

        self.in_features = in_features
        self.original_out_features = original_out_features
        self.max_original_id = max_original_id

        # initialize additional linear
        self.additional_out_features = additional_out_features
        self.has_bias = bias
        if additional_out_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=additional_out_features,
                bias=self.has_bias,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(require_regular_grad=not partially_freeze, require_additional_grad=True)

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        if self.has_bias:
            self.bias.requires_grad_(require_regular_grad)
        self.additional_fc.requires_grad_(require_additional_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        output = output[..., : self.max_original_id + 1]

        if self.additional_out_features > 0:
            additional_features = F.linear(input, self.additional_fc.weight, self.additional_fc.bias)
            output = torch.cat((output, additional_features), -1)
        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, additional_out_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.max_original_id + 1,
            self.additional_out_features,
            self.bias is not None,
            (not self.weight.requires_grad or not self.bias.requires_grad),
        )


class XGenMMVisionTokenizerConfig(PretrainedConfig):
    model_type = "xgenmm_vision_tokenizer"

    def __init__(
        self,
        vis_feature_dim: int = 1152,
        lang_embedding_dim: int = 3072,
        num_vis_tokens: int = 128,
        image_aspect_ratio: str = "anyres",
        **kwargs,
    ):
        self.vis_feature_dim = vis_feature_dim
        self.lang_embedding_dim = lang_embedding_dim
        self.num_vis_tokens = num_vis_tokens
        self.image_aspect_ratio = image_aspect_ratio
        super().__init__(**kwargs)


class XGenMMVisionTokenizer(PreTrainedModel):
    config_class = XGenMMVisionTokenizerConfig

    def __init__(self, config: XGenMMVisionTokenizerConfig):
        super().__init__(config)
        self.model = PerceiverResampler(
            dim=config.vis_feature_dim,
            dim_inner=config.lang_embedding_dim,
            num_latents=config.num_vis_tokens,
        )

    def forward(self, vision_features: torch.Tensor, vision_attn_masks: torch.Tensor):
        return self.model(vision_features, vision_attn_masks)


# copied from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/utils.py

# import torch


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def hasattr_recursive(obj, att):
    """
    Check if obj has nested attribute
    Example: hasattr_recursive(obj, 'a.b.c') is equivalent to hasattr(obj, 'a') and hasattr(obj.a, 'b') and hasattr(obj.a.b, 'c')
    """
    if att == "":
        return True
    i = att.find(".")
    if i < 0:
        return hasattr(obj, att)
    else:
        try:
            return hasattr_recursive(getattr(obj, att[:i]), att[i + 1 :])
        except:
            return False


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0) if padding_side == "right" else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


def num_params(module, filter_to_trainable=False):
    """Returns the number of parameters in the module, or optionally only the trainable parameters"""
    if filter_to_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


# copied directly from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/cross_attn_lm.py


# from .helpers import GatedCrossAttentionBlock
# from .utils import getattr_recursive, setattr_recursive


class DecoderLayerWithCrossAttention(nn.Module):
    """
    DecoderLayerWithCrossAttention is a wrapper around the GatedCrossAttentionBlock and DecoderLayer.
    """

    def __init__(self, gated_cross_attn_layer, decoder_layer, gradient_checkpointing=False):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        if self.gated_cross_attn_layer is not None:
            self.gated_cross_attn_layer._use_gradient_checkpointing = gradient_checkpointing
        self.decoder_layer._use_gradient_checkpointing = gradient_checkpointing

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None and self.media_locations is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x):
        self.vis_x = vis_x

    def condition_media_locations(self, media_locations):
        self.media_locations = media_locations

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Cross attention
        contains_media = (self.media_locations == 1).any()
        if contains_media and self.gated_cross_attn_layer is not None:
            if self.vis_x is None:
                raise ValueError("vis_x must be conditioned before forward pass")

            if self.media_locations is None:
                raise ValueError("media_locations must be conditioned before forward pass")

            lang_x = self.gated_cross_attn_layer(
                lang_x,
                self.vis_x,
                media_locations=self.media_locations,
            )

        # Normal decoder layer
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class CrossAttentionMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_cross_attention_layers(
        self,
        lang_hidden_size,
        vis_hidden_size,
        cross_attn_every_n_layers,
        gradient_checkpointing,
    ):
        """
        Add gated cross attn layers to the decoder.
        """
        old_decoder_blocks = self._get_decoder_layers()
        self.decoder_block_class = old_decoder_blocks[0].__class__
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(dim=lang_hidden_size, dim_visual=vis_hidden_size)
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(old_decoder_blocks)
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    DecoderLayerWithCrossAttention(gated_cross_attn_layer, decoder_layer, gradient_checkpointing)
                    for gated_cross_attn_layer, decoder_layer in zip(self.gated_cross_attn_layers, old_decoder_blocks)
                ]
            )
        )
        self.initialized_cross_attention = True

    def _condition_media_before_forward(
        self,
        input_ids: torch.Tensor,
        vision_tokens: torch.Tensor = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        num_beams: int = 1,
    ):
        """Each xattn layer needs to save the vision tokens and the locations of the media tokens in the language sequence"""
        assert self.initialized_cross_attention, "Cross attention layers have not been initialized. "

        # concat with past
        if past_media_locations is not None and past_vision_tokens is not None:
            if vision_tokens is not None:
                updated_vision_tokens = torch.cat(
                    [
                        past_vision_tokens,
                        vision_tokens,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = past_vision_tokens
            updated_media_locations = torch.cat(
                [
                    past_media_locations,
                    input_ids == self.media_token_id,
                ],
                dim=1,
            )
        else:
            updated_vision_tokens = vision_tokens
            updated_media_locations = input_ids == self.media_token_id

        # repeat the vision tokens and media locations for each beam
        updated_vision_tokens = updated_vision_tokens.repeat_interleave(num_beams, dim=0)
        updated_media_locations = updated_media_locations.repeat_interleave(num_beams, dim=0)

        # condition
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(updated_vision_tokens)
            layer.condition_media_locations(updated_media_locations)

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)


# copied from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/vlm.py


class VLM(nn.Module):
    """
    Generic vision-language model (VLM) class.
    A VLM consists of four components:
        1. A vision encoder that extracts features from pixels, e.g. CLIP
            input: (B, T_img, F, C, H, W)
            output: (B, T_img, F, v, d)
        2. A vision tokenizer that converts these features to visual token-like embeddings, e.g. Perceiver, or a linear projection head
            input: (B, T_img, F, v, d)
            output: (B, T_img, n, d)
        3. A fusion method that allows the language model to attend to these tokens, e.g. cross-attention, or placing the tokens directly in the language model's input sequence
        4. A language model
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): e.g. CLIP
            vision_tokenizer (nn.Module): e.g. PerceiverResampler
            lang_model (nn.Module): e.g. MPT
            initial_tokenizer_len (int): size of the original tokenizer vocab
            pad_token_id (int): id of the pad token
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # save dimension information
        self.lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        if hasattr(lang_model.config, "d_model"):
            self.lang_hidden_dim = lang_model.config.d_model  # mpt uses d_model
        else:
            self.lang_hidden_dim = lang_model.config.hidden_size
        self.vis_embedding_dim = vision_tokenizer.dim_media
        self.num_tokens_per_vis = vision_tokenizer.num_tokens_per_media

        # core components
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        lang_model = cast(Phi3ForCausalLM, lang_model)
        self.lang_model = lang_model
        if base_img_size is None:
            if isinstance(self.vision_encoder, CLIPVisionModel) or isinstance(
                self.vision_encoder, SiglipVisionTransformer
            ):
                base_img_size = self.vision_encoder.config.image_size
            else:
                base_img_size = self.vision_encoder.image_size[0]
        self.base_img_size = base_img_size

        # lm embeddings
        self.pad_token_id = pad_token_id
        self.initial_tokenizer_len = initial_tokenizer_len
        input_embeds = DecoupledEmbedding(
            max_original_id=initial_tokenizer_len - 1,
            num_additional_embeddings=len(self.special_tokens),
            _weight=self.lang_model.get_input_embeddings().weight,
            pad_token_id=self.pad_token_id,
        )
        if hasattr(input_embeds, "additional_embedding"):
            input_embeds.additional_embedding.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_input_embeddings(input_embeds)

        out_embeds = DecoupledLinear(
            max_original_id=initial_tokenizer_len - 1,
            additional_out_features=len(self.special_tokens),
            _weight=self.lang_model.get_output_embeddings().weight,
            _bias=self.lang_model.get_output_embeddings().bias
            if hasattr(self.lang_model.get_output_embeddings(), "bias")
            else None,
        )
        if hasattr(out_embeds, "additional_fc"):
            out_embeds.additional_fc.weight.data.normal_(
                mean=0.0,
                std=self.lang_model.config.initializer_range
                if hasattr(self.lang_model.config, "initializer_range")
                else 0.02,
            )
        self.lang_model.set_output_embeddings(out_embeds)

        # gradient checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor]]]] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (past_media_locations is None), (
            "past_vision_tokens and past_media_locations must both be None or both be not None"
        )

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def _encode_vision_x_anyres_custom(self, samples, device):
        assert self.anyres_grids is not None
        image_raw = samples["image"]  # list of patch list in of shape [1, N_patch, C, H, W]
        image_sizes = samples["image_size"]

        images = [x.squeeze(0) for sample_img in image_raw for x in sample_img]
        image_sizes = [s for sample_sizes in image_sizes for s in sample_sizes]

        image = torch.cat(images, dim=0)  # [\sum{B}{N_patch_i}, C, H, W]
        image = image.to(device)

        with torch.no_grad():
            image_embeds = self.vision_encoder(image, interpolate_pos_encoding=True).last_hidden_state

        grid_size_base = self.base_img_size // self.vision_encoder.config.patch_size
        grid_size = (grid_size_base, grid_size_base)
        height, width = grid_size

        if not image_embeds.shape[1] == height * width:
            assert image_embeds.shape[1] == height * width + 1  # For vision encoders that has [CLS] token.
            image_embeds = image_embeds[:, 1:, :]  # Drop the cls token for each patch.
        n_vis_token_per_patch = image_embeds.shape[1]

        # Split encoded patches and merge patch features
        # 1. Get the raw sizes from samples, and split the image embeds [\sum_{B}(N_patch_i), N_tok(16*16), C]
        split_sizes = [image.shape[0] for image in images]
        image_embeds = torch.split(image_embeds, split_sizes, dim=0)
        # 2. For each image (consist of a list of patches), merge the patches spatially (of shape [C, n_patch_height, n_patch_width])
        new_image_embeds = []
        patch_attn_masks = []
        max_n_img_token = -1
        for idx, patch_embeds in enumerate(image_embeds):
            if patch_embeds.shape[0] > 1:
                # 3. Flatten the patch features and get [C, n_patch_height * (n_patch_width+1)]
                base_patch_embeds = patch_embeds[
                    0
                ]  # TODO: prepend the CLS token for th base patch embeds (of the resized entire image).
                patch_embeds = patch_embeds[1:]

                assert height * width == base_patch_embeds.shape[0]

                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[idx], self.anyres_grids, self.base_img_size
                )  # Hardcoded grid_pinpoints.
                patch_embeds = patch_embeds.view(num_patch_height, num_patch_width, height, width, -1)

                patch_embeds = patch_embeds.permute(4, 0, 2, 1, 3).contiguous()
                patch_embeds = patch_embeds.flatten(1, 2).flatten(2, 3)
                patch_embeds, patch_attn_mask = unpad_image(patch_embeds, image_sizes[idx], self.anyres_patch_sampling)
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat(
                        (patch_embeds, self.image_newline[:, None, None].expand(*patch_embeds.shape[:-1], 1)), dim=-1
                    )
                patch_embeds = patch_embeds.view(-1, num_patch_height, num_patch_width, height * width)
                patch_embeds = patch_embeds.flatten(1, 2).permute(1, 2, 0)
                assert patch_attn_mask is not None
                patch_attn_mask = patch_attn_mask.view(num_patch_height, num_patch_width, height * width)
                patch_attn_mask = patch_attn_mask.flatten(0, 1)
                patch_embeds = torch.cat((base_patch_embeds.unsqueeze(0), patch_embeds), dim=0)
                patch_attn_mask = torch.cat(
                    (torch.ones(n_vis_token_per_patch, device=patch_embeds.device).unsqueeze(0), patch_attn_mask), dim=0
                )

            else:
                patch_embeds = patch_embeds[0].unsqueeze(0) if self.anyres_patch_sampling else patch_embeds[0]
                patch_attn_mask = (
                    torch.ones(n_vis_token_per_patch, device=patch_embeds.device).unsqueeze(0)
                    if self.anyres_patch_sampling
                    else None
                )
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat((patch_embeds, self.image_newline[None]), dim=0)

            new_image_embeds.append(patch_embeds)
            patch_attn_masks.append(patch_attn_mask)

        return new_image_embeds, patch_attn_masks

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                vision_x = self.vision_encoder.trunk.forward_features(vision_x)
            elif self.vision_encoder.__class__.__name__ in ["CLIPVisionModel", "SiglipVisionTransformer"]:
                vision_x = self.vision_encoder(vision_x).last_hidden_state
            else:
                vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _concat_vision_cache(self, lang_x, vision_tokens, past_vision_tokens, past_media_locations, use_cache):
        """
        Helper function to include the past vision tokens and past media locations in the output.
        """
        if use_cache:
            if past_media_locations is not None and past_vision_tokens is not None:
                if vision_tokens is not None:
                    updated_vision_tokens = torch.cat(
                        [
                            past_vision_tokens,
                            vision_tokens,
                        ],
                        dim=1,
                    )
                else:
                    updated_vision_tokens = past_vision_tokens
                updated_media_locations = torch.cat(
                    [
                        past_media_locations,
                        lang_x == self.media_token_id,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = vision_tokens
                updated_media_locations = lang_x == self.media_token_id

        else:
            updated_vision_tokens = None
            updated_media_locations = None

        return updated_vision_tokens, updated_media_locations

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor]]]] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        output = self.lang_model.generate(
            **new_inputs,
            past_key_values=past_key_values,
            num_beams=num_beams,
            use_cache=True,
            **kwargs,
        )
        self._post_forward_hook()
        return output

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    def set_trainable(self):
        """
        Freeze appropriate parameters in the model.
        """
        raise NotImplementedError

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if self._should_apply_weight_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
        return params_with_wd, params_without_wd

    def _should_apply_weight_decay(self, parameter_name):
        """
        Return whether weight decay should be applied to a parameter.
        """
        raise NotImplementedError

    @property
    def special_tokens(self):
        """
        Returns a dict mapping from the attribute name of a special token to its string format,
         e.g. "media_token": "<image>"
        """
        assert "media_token" in self._special_tokens, (
            "VLMs need to request that the tokenizer add a media_token and call set_special_token_ids to set self.media_token_id"
        )
        return self._special_tokens

    @property
    def special_token_ids(self):
        """
        Returns a list of the special token ids
        """
        return [getattr(self, f"{att_name}_id") for att_name in self.special_tokens]

    def set_special_token_ids(self, string_to_ids):
        """
        Args:
            string_to_ids (dict): mapping from token string to id
        """
        assert set(self.special_tokens.values()).issubset(set(string_to_ids.keys()))
        for att_name, token_str in self.special_tokens.items():
            token_id = string_to_ids[token_str]
            setattr(self, f"{att_name}_id", token_id)
            setattr(self.lang_model, f"{att_name}_id", token_id)

    def init_gradient_checkpointing(self):
        from functools import partial

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            CheckpointWrapper,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )


class VLMWithCrossAttention(VLM):
    """
    VLM using cross-attention to fuse vision and language tokens.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
        decoder_layers_attr_name: str = None,
        cross_attn_every_n_layers: int = None,
    ):
        extend_instance(lang_model, CrossAttentionMixin)
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.lang_model.set_decoder_layers_attr_name(decoder_layers_attr_name)
        self.decoder_layers_attr_name = decoder_layers_attr_name
        self.lang_model.init_cross_attention_layers(
            lang_hidden_size=self.lang_hidden_dim,
            vis_hidden_size=self.vis_embedding_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "right",  # noop for cross-attention models
        num_beams: int = 1,
    ):
        """Each xattn layer needs to save the vision tokens and the locations of the media tokens in the language sequence"""
        self.lang_model._condition_media_before_forward(
            input_ids=lang_x,
            vision_tokens=vision_tokens,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            num_beams=num_beams,
        )
        if past_key_values is not None:
            past_key_values = [
                (k.repeat_interleave(num_beams, dim=0), v.repeat_interleave(num_beams, dim=0))
                for k, v in past_key_values
            ]

        return {
            "input_ids": lang_x,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        """Include the past vision tokens and past media locations in the output"""
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        # clear the conditioned layers
        self.lang_model.clear_conditioned_layers()

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        from .helpers import GatedCrossAttentionBlock

        decoder_block_class = getattr_recursive(self.lang_model, self.decoder_layers_attr_name)[0].__class__

        def lambda_fn(module: nn.Module):
            # we want FSDP(ckpt(module)), not ckpt(FSDP(module))
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(module, CheckpointWrapper):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, GatedCrossAttentionBlock):
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        num_xattn_params = num_params(self.lang_model.gated_cross_attn_layers)
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Cross attention: {num_xattn_params:,} parameters",
                f"Language model: {num_params(self.lang_model) - num_xattn_params:,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        num_xattn_params = num_params(self.lang_model.gated_cross_attn_layers, filter_to_trainable=True)
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Cross attention: {num_xattn_params:,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True) - num_xattn_params:,} trainable parameters",
            ]
        )


class VLMWithLanguageStream(VLM):
    """
    VLM that fuses modalities by inserting vision tokens directly into the language stream.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
    ):
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            base_img_size=base_img_size,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder_layers_attr_name = decoder_layers_attr_name
        for block in getattr_recursive(self.lang_model, self.decoder_layers_attr_name):
            block._use_gradient_checkpointing = gradient_checkpointing

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "left",
        num_beams: int = 1,
    ):
        """
        Insert the vision tokens directly into the language stream/
        This requires us to modify the input_ids, attention_mask, and labels.
        """
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            assert attention_mask.shape[1] == past_len + lang_x.shape[1], (
                "Attention_mask must be as long as the entire past len (including image tokens) and current input IDs. "
                + "Check that you've expanded the attention mask to account for past image tokens."
            )

        if vision_tokens is None:
            return {
                "input_ids": lang_x,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # get the language embeddings
        lang_embeds = self.lang_model.get_input_embeddings()(lang_x)

        # build up the multimodal embeddings
        B = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None
        for i in range(B):
            # get index of <image> tokens in lang_x[i]
            image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]

            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                multimodal_attention_mask.append(attention_mask[i].clone())
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            # since an image is represented by self.num_tokens_per_vis tokens, we need to offset the image_token_idxs

            # loop through the image_token_idxs and insert the vision tokens
            new_embed = lang_embeds[i].clone()
            new_attention_mask = attention_mask[i].clone() if attention_mask is not None else None
            if has_labels:
                new_label = labels[i].clone()

            for img_num in range(len(image_token_idxs)):
                img_idx = image_token_idxs[img_num]
                # Get vision token attention mask for padded llava-style any resolution image tokens.
                if self.image_aspect_ratio == "anyres":
                    num_vis_tokens = vision_tokens[i][img_num].shape[0]
                    if vision_attention_mask is not None:
                        vis_attention_mask = vision_attention_mask[i][img_num]
                    else:
                        vis_attention_mask = torch.ones(num_vis_tokens, dtype=torch.long).to(attention_mask.device)
                else:
                    assert vision_tokens[i][img_num].shape[0] == self.num_tokens_per_vis, (
                        f"vision token number mismatch: image embedding ({vision_tokens[i][img_num].shape[0]}) \
                            vs. model.num_tokens_per_vis ({self.num_tokens_per_vis})"
                    )
                    # By default, vision tokens are not padded.
                    num_vis_tokens = self.num_tokens_per_vis
                    vis_attention_mask = torch.ones(num_vis_tokens, dtype=torch.long).to(attention_mask.device)

                # Offset the rest of image tokens with current num_vis_tokens
                for j in range(img_num + 1, len(image_token_idxs)):
                    image_token_idxs[j] += num_vis_tokens - 1

                new_embed = torch.cat(
                    (
                        new_embed[:img_idx],
                        vision_tokens[i][img_num],
                        new_embed[img_idx + 1 :],
                    ),
                    dim=0,
                )
                new_attention_mask = torch.cat(
                    (
                        new_attention_mask[:img_idx],
                        vis_attention_mask,
                        new_attention_mask[img_idx + 1 :],
                    ),
                    dim=0,
                )
                if has_labels:
                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(num_vis_tokens, dtype=torch.long).to(labels.device) * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )
            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)

        # stack
        multimodal_embeds = stack_with_padding(
            multimodal_embeds,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        multimodal_attention_mask = stack_with_padding(
            multimodal_attention_mask,
            padding_value=0,
            padding_side=padding_side,
        )
        if has_labels:
            multimodal_labels = stack_with_padding(
                multimodal_labels,
                padding_value=-100,
                padding_side=padding_side,
            )

        return {
            "inputs_embeds": multimodal_embeds,
            "attention_mask": multimodal_attention_mask,
            "labels": multimodal_labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        # Include the past vision tokens and past media locations in the output
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )

        # return logits that are the same shape as the original input_ids
        logits = output.logits
        batch_logits = []
        B, T_txt = lang_x.shape
        for i in range(B):
            sequence_logits = []
            logits_j = 0
            img_id = 0
            for j in range(T_txt):
                if lang_x[i, j] != self.media_token_id:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    # logits_j += self.num_tokens_per_vis
                    # Offset in account of dynamic num_vis_tokens.
                    logits_j += vision_tokens[i][img_id].shape[0]
                    img_id += 1
            sequence_logits = torch.stack(sequence_logits, dim=0)  # (B, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(batch_logits, dim=0)  # (B, T_txt, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == (B, T_txt)

        # assemble the output
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=batch_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        pass

    def get_fsdp_lambda_fn(self):
        """
        Returns the lambda function used to decide how to perform FSDP wrapping.
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        decoder_block_class = getattr_recursive(self.lang_model, self.decoder_layers_attr_name)[0].__class__

        def lambda_fn(module: nn.Module):
            if getattr(module, "_use_gradient_checkpointing", False) and not isinstance(module, CheckpointWrapper):
                return False
            if module is self.vision_tokenizer:
                return True
            if isinstance(module, decoder_block_class):
                return True

        return lambda_fn

    def get_fsdp_wrapping_policy(self):
        """
        Returns the policy used to decide how to perform FSDP wrapping.
        """
        from open_clip.transformer import ResidualAttentionBlock, VisionTransformer
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.phi.modeling_phi import PhiDecoderLayer

        # for Phi-3 hot fiix
        try:
            import importlib

            commit_hash = str(type(self.lang_model)).split("instruct.")[1].split(".modeling")[0]
            module_name = f"transformers_modules.microsoft.Phi-3-mini-128k-instruct.{commit_hash}.modeling_phi3"
            module = importlib.import_module(module_name)
            Phi3DecoderLayer = module.Phi3DecoderLayer
            import_phi3 = True
        except IndexError:
            import_phi3 = False

        # hard code the wrap module name
        # vision
        if isinstance(self.vision_encoder, SiglipVisionModel):
            from transformers import SiglipVisionModel

            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={SiglipVisionModel})
            from transformers.models.siglip.modeling_siglip import (
                SiglipEncoderLayer,
                SiglipMultiheadAttentionPoolingHead,
                SiglipVisionEmbeddings,
                SiglipVisionTransformer,
            )

            # import torch.nn.LayerNorm as LayerNorm
            transformer_layer_cls_vit = {
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                SiglipVisionEmbeddings,
                SiglipMultiheadAttentionPoolingHead,
            }
            vision_transformer_block_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit
            )
            vision_wrap_policy = functools.partial(
                _or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy]
            )

        else:
            vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer, TimmModel})
            # vit_wrap_policy = functools.partial(_module_wrap_policy, module_classes={VisionTransformer})
            # transformer_layer_cls_vit = {ResidualAttentionBlock}
            transformer_layer_cls_vit = {ResidualAttentionBlock, Block}
            # transformer_layer_cls_vit = {Block}
            vision_transformer_block_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls_vit
            )
            vision_wrap_policy = functools.partial(
                _or_policy, policies=[vit_wrap_policy, vision_transformer_block_policy]
            )
        # llm
        transformer_layer_cls = {LlamaDecoderLayer, PhiDecoderLayer}
        if import_phi3:
            transformer_layer_cls.add(Phi3DecoderLayer)
        llm_transformer_block_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_cls
        )
        # vision_tokenizer
        vis_tokenizer_policy = functools.partial(
            _module_wrap_policy, module_classes={LinearPatchProjection, PerceiverResampler}
        )
        return functools.partial(
            _or_policy, policies=[vision_wrap_policy, llm_transformer_block_policy, vis_tokenizer_policy]
        )

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Language model: {num_params(self.lang_model):,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True):,} trainable parameters",
            ]
        )


# copied from https://github.com/salesforce/LAVIS/blob/xgen-mm/open_flamingo/src/xgenmm.py with some changes


# from .helpers import PerceiverResampler
# from .vlm import VLMWithLanguageStream


class XGenMMPerceiver(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        base_img_size: Optional[int] = None,
        image_aspect_ratio: str = "anyres",
        anyres_patch_sampling: bool = True,
        num_vision_tokens: int = 128,
        anyres_grids: list[int] = None,
        max_num_frames: int | None = None,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
            max_num_frames (int, optional): time embedding dimension in PerceiverResampler (vision tokenizer). Defaults to 0.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(
                dim=vis_feature_dim,
                dim_inner=lang_embedding_dim,
                num_latents=num_vision_tokens,
                max_num_frames=max_num_frames,
            ),
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            base_img_size=base_img_size,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling

        # https://github.com/huggingface/peft/issues/1827
        self.prepare_inputs_for_generation = self.lang_model.prepare_inputs_for_generation
        self.config = self.lang_model.config
        # self.anyres_grids = None
        self.anyres_grids = anyres_grids

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True

    def forward(
        self,
        vision_x: list[list[torch.Tensor]],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor]]]] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (past_media_locations is None), (
            "past_vision_tokens and past_media_locations must both be None or both be not None"
        )
        # convert pixels to vision tokens
        vision_attention_mask = None

        input_dict = dict(image=vision_x, image_size=image_size)
        vision_features, vision_attn_masks = self._encode_vision_x_anyres_custom(input_dict, lang_x.device)

        split_sizes = [feature.shape[0] for feature in vision_features]
        nt_images = [len(images) for images in vision_x]
        split_split_sizes = []
        img_id = 0
        for nt in nt_images:
            split_split_sizes.append(split_sizes[img_id : img_id + nt])
            img_id += nt

        vision_features = torch.cat(vision_features, dim=0)
        vision_features = vision_features[:, None, None, :, :]  # Expand dimensions.
        vision_attn_masks = torch.cat(vision_attn_masks, dim=0)

        vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks)

        vision_token_groups = torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
        vision_tokens = []

        for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
            patch_vis_token_groups = torch.split(
                patch_vis_tokens, split_split_sizes[sample_id], dim=0
            )  # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
            flatten_vision_tokens = []
            # padded_attn_masks = []
            for image_vis_token in patch_vis_token_groups:
                image_vis_token = image_vis_token.flatten(0, 2)  # [Np, 1, v, d] -> [Np*v, d]
                flatten_vision_tokens.append(image_vis_token)
            vision_tokens_i = flatten_vision_tokens
            vision_tokens.append(vision_tokens_i)

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )
        # postforward hooks
        # self._post_forward_hook()
        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        image_size: Optional[Tuple] = None,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[List[Union[torch.Tensor, Tuple[torch.Tensor]]]] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == "anyres":
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(input_dict, lang_x.device)
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id : img_id + nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[:, None, None, :, :]  # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, vision_attn_masks)

            if (
                not self.anyres_patch_sampling and self.image_aspect_ratio == "anyres"
            ):  # copied to here so the main loop does not break
                split_sizes = [feature.shape[0] for feature in vision_features]
                # Nested splits for multi-image samples.
                if isinstance(vision_x[0], list):
                    nt_images = [len(images) for images in vision_x]
                    split_split_sizes = []
                    img_id = 0
                    for nt in nt_images:
                        split_split_sizes.append(split_sizes[img_id : img_id + nt])
                        img_id += nt
                else:
                    nt_images = [1] * len(vision_x)
                    split_split_sizes = split_sizes

                vision_tokens = vision_tokens.squeeze()
                vision_tokens = list(
                    torch.split(vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0)
                )  # [batch], frame, vistok, embed

            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                assert isinstance(vision_x, list)
                if isinstance(vision_x[0], list):
                    vision_token_groups = torch.split(
                        vision_tokens, list(sum(nt_img) for nt_img in split_split_sizes), dim=0
                    )
                    vision_tokens = []

                    for sample_id, patch_vis_tokens in enumerate(vision_token_groups):
                        # Pad the image tokens within a sample.
                        patch_vis_token_groups = torch.split(
                            patch_vis_tokens, split_split_sizes[sample_id], dim=0
                        )  # [Np*nt, 1, v, d] -> [[Np_t, 1, v, d], ...]
                        flatten_vision_tokens = []
                        for image_vis_token in patch_vis_token_groups:
                            image_vis_token = image_vis_token.flatten(0, 2)  # [Np, 1, v, d] -> [Np*v, d]
                            flatten_vision_tokens.append(image_vis_token)
                        vision_tokens_i = flatten_vision_tokens
                        vision_tokens.append(vision_tokens_i)
                else:
                    # Padding. FIXME: padding here might not be necessary?
                    vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                    # Padding.
                    vision_tokens = []
                    for patch_vis_tokens in vision_token_groups:
                        patch_vis_tokens = patch_vis_tokens.flatten(0, 2)  # [Np, 1, v, d] -> [Np*v, d]
                        vision_tokens.append(patch_vis_tokens.unsqueeze(0))  # Add the nt dimension.
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        if past_key_values is not None:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output
