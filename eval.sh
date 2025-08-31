model_path="./models/RSMP_eval"
output_path="./outputs/RSMP_eval"
log_path="./logs/RSMP_eval"
model_file="./models/RSMP/model_seed_0_Iter_3.pkl"

CUDA_VISIBLE_DEVICES=0,1 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}