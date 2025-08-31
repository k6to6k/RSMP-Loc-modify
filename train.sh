model_path="./models/RSMP"
output_path="./outputs/RSMP"
log_path="./logs/RSMP"
seed=0

CUDA_VISIBLE_DEVICES=0,1 python -W ignore ./main.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --seed ${seed}