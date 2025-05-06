export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=./models/JudgeLM-7B
BASE_MODEL_PATH=./models/vicuna-7b
MODEL_TYPE=judgelm
DATA_TYPE=pandalm

python3 -u src/cal_reliability.py \
    --model-name-or-path $MODEL_PATH \
    --cali-model-name-or-path $BASE_MODEL_PATH \
    --model-type ${MODEL_TYPE} \
    --data-type ${DATA_TYPE} \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

python -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"