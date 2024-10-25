# How to reproduce our results?


## RQ1: Do the manually classified comments contribute to an improvement in the detection of SATD?

We train each model using [Maldonaldo](https://github.com/maldonado/tse.satd.data) dataset and with extra Tesoro data. We then evaluate the performance improvement achieved by integrating the Tesoro data.


**Training scripts:** See more in [scripts/RQ1](scripts/RQ1)

```bash
MODEL_SHORT_NAME=codebert-base
MODEL_NAME=microsoft/codebert-base
TASK={detection|classification|identification}

python3 cross_validation.py \
--seed 0 \
--model_short_name $MODEL_SHORT_NAME \
--model_name_or_path $MODEL_NAME \
--train_file ../data/kfolds/maldonado62k \
--output_dir ../results/maldonado62k/$MODEL_SHORT_NAME-satd-$TASK \
--text_column_names comment \
--label_column_name classification \
--metric_for_best_model f1 \
--metric_name f1 \
--max_seq_length 256 \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--do_train \
--do_predict \
--load_best_model_at_end \
--evaluation_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--save_total_limit 1 \
--overwrite_output_dir \
--cross_validation \
--fp16 \
# --bin_classification \ # Use for SATD identification scenario
# --ignore_label WITHOUT_CLASSIFICATION \ # Use for SATD classification scenario
# --extra_file ../data/tesoro_as_extra_data.json \ # Use for incoporating Tesoro
```

## RQ2: Do the manually classified comments contribute to an improvement in the detection of SATD?
