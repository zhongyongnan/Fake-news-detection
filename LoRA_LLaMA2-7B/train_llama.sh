#!/bin/bash

set -e 
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Model Path
model_name=llama-2-7b-chat-hf
model_path=/data/Logic/${model_name}
tokenizer=${model_path}

# Data Path
data_path=/data1/mwy/NLP/LoRA_LLaMA2-7B/train.jsonl

# Output Path
output_path=/data1/mwy/NLP/${model_name}_fnc #/path/to/your/output/

#Save LoRA
lora_path=${output_path}/lora
mkdir -p ${lora_path}/

# Deepspeed
ds_config_file=/data/Logic/train_scripts/deepspeed_configs/ds_config_stage3.json

# Train Parameter
bs_per_gpu=8
num_nodes=1
nproc_per_node=`nvidia-smi | grep MiB | wc -l`
master_port=50005

# grad_acc=`expr 256 / ${bs_per_gpu} / ${num_nodes} / ${nproc_per_node}`
grad_acc=1
deepspeed --include 'localhost:0' --master_port ${master_port} train.py \
    --model_name_or_path ${model_path} \
    --tokenizer ${tokenizer} \
    --data_path ${data_path} \
    --output_dir ${lora_path} \
    --per_device_train_batch_size ${bs_per_gpu} \
    --gradient_accumulation_steps ${grad_acc} \
    --bf16 True \
    --gradient_checkpointing_enable True \
    --num_train_epochs 2 \
    --model_max_length 2048 \
    --learning_rate 2.5e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps -1 \
    --save_total_limit 999 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --enable_lora True \
    --deepspeed ${ds_config_file} | tee ${output_path}/training_log.txt

# Convert lora to huggingface model
CUDA_VISIBLE_DEVICES="0" python convert_to_hf.py \
     --model_name_or_path ${model_path} \
     --lora_path ${lora_path}   \
     --output_dir ${output_path}