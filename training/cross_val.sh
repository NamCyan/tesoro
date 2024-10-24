#! /bin/bash

# comment - 256, code - 512
# my dataset batch 16

CUDA_VISIBLE_DEVICES=0,1 python3 cross_validation.py \
--seed 0 \
--model_short_name graphcodebert-comment-att \
--model_name_or_path microsoft/graphcodebert-base \
--train_file ../Data/My_dataset/BATCH1/kfolds_0.0.1 \
--output_dir ../results/my-dataset-0.0.1/graphcodebert-kfold_comment-code_context_2-comment-att-identification \
--text_column_names comment,code \
--label_column_name classification \
--metric_for_best_model f1 \
--metric_name f1 \
--text_column_delimiter "</s></s>" \
--max_seq_length 256 \
--max_query_length 512 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
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
--bin_classification \
--remove_special_tokens \
# --ignore_label WITHOUT_CLASSIFICATION \
# --is_llm \
# --lora_config_path lora_config/lora_config.yaml \
# --low_cpu_mem_usage \
# --device_map auto \
# --gradient_checkpointing \
# --is_enc_dec \
# --extra_file ../Data/My_dataset/BATCH1/tesoro_as_extra_data.json \
# --ignore_label WITHOUT_CLASSIFICATION \
# --bin_classification \
