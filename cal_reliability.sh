export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="judgelm"
DATA_TYPE="salad-bench"

#python -u src/cal_reliability.py \
#    --model-name-or-path "./models/JudgeLM-7B" \
#    --cali-model-name-or-path "./models/vicuna-7b/" \
#    --model-type ${MODEL_TYPE} \
#    --data-type $DATA_TYPE \
#    --max-new-token 1024 \
#    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
#    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

python -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"