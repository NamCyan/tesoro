#! /bin/bash
MODEL_SHORT_NAME=codebert-base
MODEL_NAME=microsoft/codebert-base
CODE_CONTEXT=code_context_2
TECHNIQUE=concat

if [ $TECHNIQUE != 'concat' ]; then
	MODEL_SHORT_NAME=$MODEL_SHORT_NAME-comment-att
fi

python3 cross_validation.py \
--seed 0 \
--model_short_name $MODEL_SHORT_NAME \
--model_name_or_path $MODEL_NAME \
--train_file ../data/kfolds/maldonado62k \
--output_dir ../results/maldonado62k/$MODEL_SHORT_NAME-$CODE_CONTEXT-$TECHNIQUE \
--text_column_names comment,$CODE_CONTEXT \
--label_column_name classification \
--metric_for_best_model f1 \
--metric_name f1 \
--max_seq_length 256 \
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
