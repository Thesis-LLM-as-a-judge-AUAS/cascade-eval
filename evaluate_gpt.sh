#!/bin/bash
DATA_TYPE="salad-bench"
MODEL_NAME="gpt-3.5-turbo"

python -u src/evaluate_gpt.py \
    --model-name $MODEL_NAME \
    --prompt-type "vanilla" \
    --data-type $DATA_TYPE \
    --multi-process True \
    --max-new-token 1024 \
    --rewrite-output True