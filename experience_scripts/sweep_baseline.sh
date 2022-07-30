#!/bin/bash

read -p "cuda: " CUDA
read -p "models: " models
read -p "dir: " dir
read -p "num_layers: " num_layers
read -p "task: " task

datasets="squirrel chameleon ACTOR TEXAS WISCONSIN CORNELL"
val='baseline'
for m in $models; do
    for dataset in $datasets; do
        wait; python main.py --cuda_num="$CUDA" --compare_model=1 --type_model "$m"  --dataset "$dataset" --num_layers "$num_layers" --N_exp 10 --task "$task"  --log_dir "$dir/$dataset/$m/$val" \

        done
    done
