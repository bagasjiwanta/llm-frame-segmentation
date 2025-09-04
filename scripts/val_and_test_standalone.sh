#!/bin/bash

# Use today's date as if not provided
if [[ -z "$1" ]]; then
    experiment_name=$(date +"%m-%d")_$(date +"%H-%M") 
else
    experiment_name=${1}
fi

if [[ ! -e runs/$experiment_name ]]; then
    mkdir -p runs/$experiment_name
fi

export PYTHONPATH="."  # training without having to pip install
export NCCL_P2P_DISABLE=1  # comment out if nvlink is present
# export TORCH_LOGS="+dynamo"

wandb_project="blip3-mr"
wandb_entity="bagas-jiwanta"


args=(
    # --vision_tokenizer_train            # Finetune the vision tokenizer module
    # --lang_model_lora                   # Use LoRA for the language model
    # --lang_model_pretrained jwnt4/xgenmm-mr-v1-lang_model-pissa-r16a16-rslora
    # --vision_tokenizer_pretrained "./pretrained/xgenmm-mr-v1-vision_tokenizer/vision_tokenizer.safetensors"
    # --lora_r 16
    # --lora_dropout 0.025
    # --init_lora_weights pissa_niter_4
    --training_precision bf16           
    --base_model_name_or_path jwnt4/xgenmm-mr-v1-merged
    # --gradient_checkpointing            

    --base_data_dir datasets            # base data dir
    --dataset_config config.yaml        # yaml config file name in base_data_dir
    --dataset_name main                 # name of dataset in base_data_dir/dataset_config file (yaml)
    --test_dataset_name test            # optional
    --num_val_workers 8                 # val dataloader workers
    --sampler pytorch                # use stratified sampler so every batch has hard, medium, and easy samples

    --val_batch_size 6
    --do_val                            # do validation
    --do_test                           # do test
    # --num_val_samples 60                   # sample this amount instead all
    # --num_test_samples 60                   # sample this amount instead all
    --num_val_beams 2                   # beam search

    --monitor "R1-at-0.7"               # monitored val (for pretty print and checkpoint save directory)
    --run_name "${experiment_name}"     # Pass the experiment name to the script

    --seed 2109
    --extra_verbose
)

# python -m debugpy --listen 5678 --wait-for-client -m deepspeed.launcher.runner \
python blip3_mr/finetune.py \
    "${args[@]}" \
    2>&1 | tee "runs/${experiment_name}/terminal_output.log"

