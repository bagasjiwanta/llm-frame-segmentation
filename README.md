# BLIP-3 MR

Moment retrieval with BLIP-3.

## Links

Original model: [Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5)

Original dataset: [https://github.com/jayleicn/moment_detr](https://github.com/jayleicn/moment_detr) 

Created dataset (train & val): [jwnt4/qvhighlights-25frames](https://huggingface.co/datasets/jwnt4/qvhighlights-25frames) 

Created dataset (test): [jwnt4/qvhighlights-25frames-test](https://huggingface.co/datasets/jwnt4/qvhighlights-25frames-test)

## Version 1 (11 epoch)

### Config
```
- train_micro_batch_size_per_gpu: 16
- gpu: 2x H100
- gradient_accumulation_steps: 4
- learning_rate: 2e-05
- warmup_steps: 282 (6 epoch)
- weight of cross entropy loss (all logits): 0.2
- weight of binary cross entropy loss (frame length): 0.2666
- weight of tversky loss (frame length): 0.2666
- weight of generalized dice loss (frame length): 0.2666
- tversky loss beta: 0.7
- binary cross entropy pos_weight: 2.3378
- bits: bf16
```

Peft:
```
LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.025,
    bias="none",
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "gate_up_proj", "qkv_proj", "gate_down_proj"],
    use_rslora=True,
    init_lora_weights="pissa_niter_4",
    task_type="CAUSAL_LM",
)
```
### QVHighlights codalab scores
```
- test_MR-full-R1@0.5: 60.77
- test_MR-full-R1@0.7: 37.42
- test_MR-full-mAP: 35.28
- test_MR-full-mAP@0.5: 56.26
- test_MR-full-mAP@0.75: 35.80
- test_MR-long-mAP: 53.79
- test_MR-middle-mAP: 29.77
- test_MR-short-mAP: 1.29
- test_HL-min-VeryGood-mAP: 34.48
- test_HL-min-Good-mAP: 59.31
- test_HL-min-Good-Hit1: 71.98
- test_HL-min-Fair-mAP: 72.18
- test_HL-min-VeryGood-Hit1: 56.74
- test_HL-min-Fair-Hit1: 75.62
- val_MR-full-R1@0.5: 62.32
- val_MR-full-R1@0.7: 38.19
- val_MR-full-mAP: 36.17
- val_MR-full-mAP@0.5: 56.80
- val_MR-full-mAP@0.75: 37.22
- val_MR-long-mAP: 52.64
- val_MR-middle-mAP: 32.64
- val_MR-short-mAP: 1.11
- val_HL-min-VeryGood-mAP: 34.64
- val_HL-min-Good-mAP: 59.20
- val_HL-min-Good-Hit1: 72.13
- val_HL-min-Fair-mAP: 71.78
- val_HL-min-VeryGood-Hit1: 58.52
- val_HL-min-Fair-Hit1: 74.90
```
### Links

Merged model: [jwnt4/xgenmm-mr-v1-merged](https://huggingface.co/jwnt4/xgenmm-mr-v1-merged)

Vision tokenizer weights: [jwnt4/xgenmm-mr-v1-vision_tokenizer](https://huggingface.co/jwnt4/xgenmm-mr-v1-vision_tokenizer)

Language model adapter: [jwnt4/xgenmm-mr-v1-lang_model-pissa-r16a16-rslora](https://huggingface.co/jwnt4/xgenmm-mr-v1-lang_model-pissa-r16a16-rslora)


## Setup

This repository was originally built using [Modal](https://modal.com).

### System Requirements

- Ubuntu 24.04 recommended
- CUDA-compatible GPU (bfloat 16 support is needed)
- Python 3.13
- CUDA 12.9.1 with cuDNN

## Install System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-setuptools \
    make \
    cmake \
    pkg-config \
```

### Install Python Packages

Upgrade essentials

```bash
pip install --upgrade pip wheel setuptools ninja numpy
```

Install torch

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

Install requirements using requirements.txt (requirements snapshot in requirements_snapshot.tx):

```bash
pip install -r requirements.txt
```

Requirements:

```
einops
einops_exts
sentencepiece
protobuf
transformers
accelerate
scikit-learn
tqdm
wandb
huggingface-hub[cli]
pillow
peft
deepspeed
```

Flash attention

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp313-cp313-linux_x86_64.whl
```
