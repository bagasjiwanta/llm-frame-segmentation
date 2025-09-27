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
    --lang_model_lora                   # Use LoRA for the language model
    --use_local_model 
    # --lang_model_pretrained jwnt4/xgenmm-mr-v1-lang_model-pissa-r16a16-rslora
    # --vision_tokenizer_pretrained "./pretrained/xgenmm-mr-v1-vision_tokenizer/vision_tokenizer.safetensors"
    --lora_r 16
    --lora_dropout 0.025
    --init_lora_weights pissa_niter_4
    --use_rslora
    --training_precision bf16           
    # --base_model_name_or_path Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5
    --gradient_checkpointing            # Saves VRAM
    --gradient_accumulation_steps 4

    --base_data_dir datasets            # base data dir
    --dataset_config config.yaml        # yaml config file name in base_data_dir
    --dataset_name main                 # name of dataset in base_data_dir/dataset_config file (yaml)
    --test_dataset_name test            # optional
    --num_train_workers 6               # train dataloader workers
    --num_val_workers 4               # val dataloader workers
    --sampler pytorch                # use stratified sampler so every batch has hard, medium, and easy samples

    --train_micro_batch_size_per_gpu 4 # train batch per gpu
    --do_train                          # do training
    --num_epochs 20                     
    --learning_rate 2e-5                
    --weight_decay 0.005
    --warmup_steps 282

    # --ce_pos_weight 1.0                 # positive weight for the cross entropy loss
    --bce_pos_weight 2.3378             # positive weight for the bce loss
    --gd_norm square                    # norm used for generalized dice ('square' or 'linear')
    --tvl_beta 0.7                      # weight on recall
    --loss_mapping_path loss_mapping.yaml  # loss weights per epoch 
    --soft_loss                         # use soft labels

    --deepspeed_config deepspeed_configs/zero2_offload.json # config json path
    --deepspeed                         # enable deepspeed

    --val_batch_size 3
    --do_val                            # do validation
    --do_test                           # do test
    --float_sanity_epoch 0.03           # percentage of validation data to be done before training
    --num_val_samples 1550                # total sample number
    --num_val_beams 2                   # beam search
    --num_val_per_epoch 1               # how many validation done in a single epoch

    --monitor "R1-at-0.7"               # monitored val (for pretty print and checkpoint save directory)
    --logging_steps 4                   
    --checkpoint_every_n_val 1          # how many validation per checkpoint
    --checkpoint_dir "/checkpoints/blip3-mr"    # checkpoint directory
    --run_name "${experiment_name}"     # Pass the experiment name to the script
    --resume_from_latest                # resume from latest checkpoint
    # --resume_from_checkpoint global_step506_R1-at-0.7-43.1500

    --seed 2109
    # --report_to_wandb 
    --wandb_project ${wandb_project} 
    --wandb_entity ${wandb_entity} 
)

# python -m debugpy --listen 5678 --wait-for-client -m deepspeed.launcher.runner \
deepspeed \
    --num_nodes 1 --num_gpus 2 \
    blip3_mr/finetune.py \
    "${args[@]}" \
    2>&1 | tee "runs/${experiment_name}/terminal_output.log"

