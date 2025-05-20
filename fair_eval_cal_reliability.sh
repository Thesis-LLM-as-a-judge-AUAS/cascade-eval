export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=./models/judgelm-7b
BASE_MODEL_PATH=./models/vicuna-7b
MODEL_TYPE=judgelm
DATA_TYPE=vicuna-mec


for i in $(seq 1 47); do
  python3 -u src/cal_reliability.py \
      --model-name-or-path $MODEL_PATH \
      --cali-model-name-or-path $BASE_MODEL_PATH \
      --model-type ${MODEL_TYPE} \
      --data-type ${DATA_TYPE} \
      --max-new-token 1024 \
      --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-logit.jsonl" \
      --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}/${i}-relia.json"

done