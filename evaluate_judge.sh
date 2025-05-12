#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
DATA_TYPE="salad-bench"
MODEL_TYPE="judgelm"

python3 -u src/evaluate_judge.py \
    --model-name-or-path "./models/Auto-J-13b" \
    --logit-file "./relia_scores/$MODEL_TYPE/$DATA_TYPE-logit.jsonl" \
    --prompt-type "vanilla" \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --max-new-token 1024