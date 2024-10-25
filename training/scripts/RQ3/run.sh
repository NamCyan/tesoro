#! /bin/bash
MODEL_SHORT_NAME=codebert-base
MODEL_NAME=microsoft/codebert-base

python3 cross_validation.py \
--seed 0 \
--model_short_name $MODEL_SHORT_NAME \
--model_name_or_path $MODEL_NAME \
--train_file ../data/kfolds/tesoro_code \
--output_dir ../results/tesoro_code/$MODEL_SHORT_NAME \
--text_column_names cleancode \
--label_column_name label \
--metric_for_best_model f1 \
--metric_name f1 \
--max_seq_length 512 \
--max_query_length 512 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 20 \
--do_train \
--do_predict \
--load_best_model_at_end \
--evaluation_strategy steps \
--eval_steps 100 \
--save_steps 100 \
--save_total_limit 1 \
--overwrite_output_dir \
--cross_validation \
--fp16 \
# arguments to train LLMs
# --is_llm \
# --lora_config_path lora_config/lora_config.yaml \
# --low_cpu_mem_usage \
# --device_map auto \
