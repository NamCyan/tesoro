#!/bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh35/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh35/.cache"
export https_proxy=http://10.16.29.10:8080


CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --model_name_or_path roberta-base \
    --train_file /home/namlh31/project/AI4Code/SATD/data/concat_data.json \
    --seed 42 \
    --balance_data True \
    --pad_to_max_length False \
    --validation_rate 0.02 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 100 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 2 \
    --input_column comment \
    --label_column classification \
    --max_seq_length 256 \
    --output_dir /cm/archive/namlh35/backup/AI4Code/SATD/exps/roberta-WITHOUT_CLASSIFICATION \
    --cache_dir /cm/archive/namlh35/.cache \
    --do_train \
    --do_eval \
    --focus_label WITHOUT_CLASSIFICATION \
    --num_labels 2 \
    --overwrite_output_dir \