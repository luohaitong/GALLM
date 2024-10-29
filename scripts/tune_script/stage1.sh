# to fill in the following path to run the first stage of our GALLM!

export PYTHONPATH="/path/to/GALLM"

dataset=cora
task=tm
id=tm
template=nd
model_path=~/lmsys/vicuna-7b-v1.5-16k
data_path_prefix=dataset
pretra_gnn=pretrained_gnn/${dataset}_gt/
output_model=checkpoints/${dataset}/${id}

wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=20004 \
    gallm/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --use_dataset ${dataset} \
    --use_task ${task} \
    --data_path_prefix ${data_path_prefix} \
    --graph_tower ${pretra_gnn} \
    --label_iaware False \
    --template ${template} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb