# to fill in the following path to evaluate!

export PYTHONPATH="/path/to/GALLM"

dataset=cora
task=nc_mp
id=nc_mp
template=nd

start_id=0
end_id=-1
num_gpus=8

output_model=checkpoints/${dataset}/${id}
pretra_gnn=pretrained_gnn/${dataset}_gt/
res_path=output_res/${dataset}/${id}
path_prefix=dataset


python ./gallm/eval/run_graphgpt.py \
--model-name ${output_model} \
--pretra_gnn ${pretra_gnn} \
--output_res_path ${res_path} \
--start_id ${start_id} \
--end_id ${end_id} \
--num_gpus ${num_gpus} \
--dataset ${dataset} \
--task ${task} \
--path_prefix ${path_prefix} \
--template ${template}