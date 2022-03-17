#!/bin/bash

read -p "cuda: " CUDA
read -p "models: " models
read -p "dir: " dir
read -p "dataset: " datasets
read -p "--compare_model? " trick

tasks="edgeMask contrastive attrMask VGAE"
#datasets="ACTOR TEXAS WISCONSIN CORNELL AmazonComputers Citeseer Cora"

declare -A dim_hidden=( ['WISCONSIN']="64" ["CORNELL"]="32" ["ACTOR"]="16" ["TEXAS"]="16" ["Citeseer"]="16" ["Cora"]="16" ["AmazonComputers"]="16" )
declare -A prompt_dim_hidden=( ['WISCONSIN']="64" ["CORNELL"]="64" ["ACTOR"]="64" ["TEXAS"]="64" ["Citeseer"]="256" ["Cora"]="128" ["AmazonComputers"]="256" )
declare -A prompt_k=( ['WISCONSIN']="5" ["CORNELL"]="5" ["ACTOR"]="5" ["TEXAS"]="5" ["Citeseer"]="3" ["Cora"]="3" ["AmazonComputers"]="3")
declare -A alpha=( ['WISCONSIN']="0.7" ["CORNELL"]="0.7" ["ACTOR"]="0.7" ["TEXAS"]="0.7" ["Citeseer"]="0.1" ["Cora"]="0.1" ["AmazonComputers"]="0.1" )
declare -A lr=( ['WISCONSIN']="0.02" ["CORNELL"]="0.02" ["ACTOR"]="0.02" ["TEXAS"]="0.02" ["Citeseer"]="0.006" ["Cora"]="0.006" ["AmazonComputers"]="0.006" )
declare -A cont=( ['WISCONSIN']="True" ["CORNELL"]="True" ["ACTOR"]="False" ["TEXAS"]="False" ["Citeseer"]="False" ["Cora"]="True" ["AmazonComputers"]="False" )

for val in $tasks; do
  type_model="GCN"
  if [ "$val" == "VGAE" ]; then
    type_model="VGAE"
  fi
  for m in $models; do
    for dataset in $datasets; do
        layers="2"
        if [ "$m" == "GCNII" ]; then
          layers="32"
        fi

        if [ "$dataset" == "ACTOR" ]; then
          epochs="120"
        elif [ "$dataset" == "AmazonComputers" ]; then
          epochs="480"
        else
          epochs="1000"
        fi

        wait; python main.py --cuda_num="$CUDA" --compare_model="$trick" \
        --dataset "$dataset" \
        --prompt-dim-hidden "${prompt_dim_hidden[$dataset]}"\
        --dim_hidden "${dim_hidden[$dataset]}" \
        --alpha "${alpha[$dataset]}" \
        --embedding_dropout 0.3 \
        --epochs "$epochs"\
        --lr "${lr[$dataset]}" \
        --num_layers "$layers" \
        --prompt-aggr edges \
        --prompt-distance-temp 1.0 \
        --prompt-head "$m" \
        --prompt-k "${prompt_k[$dataset]}" \
        --prompt-layer 3 \
        --prompt-lr 0.2 \
        --prompt-pretrain-lr 0.03 \
        --prompt-pretrain-type "$val" \
        --prompt-temp 2.0\
        --prompt-type macmip \
        --type_model "$type_model" \
        --N_exp 100  \
        --log_dir "$dir/$dataset/$m/$val" \
        --prompt-continual "${cont[$dataset]}"

        done
    done
done