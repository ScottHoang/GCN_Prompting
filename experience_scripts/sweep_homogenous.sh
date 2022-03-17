#!/bin/bash

read -p "cuda: " CUDA
read -p "models: " models
read -p "dir: " dir

tasks="VGAE edgeMask contrastive attrMask"
datasets="Citeseer Cora"


for val in $tasks; do
  type_model="GCN"
  if [ "$val" == "VGAE" ]; then
    type_model="VGAE"
  fi
  for m in $models; do
    for dataset in $datasets; do
        layers="2"
        extra=""
        if [ "$m" == "GCNII" ]; then
          layers="32"
        fi
        if [ "$dataset" == "Cora" ]; then
          hidden="64"
        else
          hidden='256'
        fi

        if [ "$dataset" == "ACTOR" ]; then
          epochs="120"
        elif [ "$dataset" == "AmazonComputers" ]; then
          epochs="480"
        else
          epochs="1000"
        fi



        if [[ -z $extra ]]; then
          wait; python main.py --cuda_num="$CUDA" --compare_model=1 --dataset "$dataset" --prompt-dim-hidden "$hidden" --alpha 0.7 \
          --embedding_dropout 0.3 --epochs "$epochs" --lr 0.02 --num_layers "$layers" --prompt-aggr edges \
          --prompt-distance-temp 1.0 --prompt-head "$m" --prompt-k 1 --prompt-layer 3 --prompt-lr 0.2 \
          --prompt-pretrain-lr 0.05 --prompt-pretrain-type "$val" --prompt-temp 1.0\
          --prompt-type macmip --type_model "$type_model" --N_exp 100  --log_dir "$dir/$dataset/$m/$val"
        else
          wait; python main.py --cuda_num="$CUDA" --compare_model=1 --dataset "$dataset" --prompt-dim-hidden "$hidden" --alpha 0.7 \
          --embedding_dropout 0.3 --epochs "$epochs" --lr 0.02 --num_layers "$layers" --prompt-aggr edges \
          --prompt-distance-temp 1.0 --prompt-head "$m" --prompt-k 1 --prompt-layer 3 --prompt-lr 0.2 \
          --prompt-pretrain-lr 0.05 --prompt-pretrain-type "$val" --prompt-temp 1.0\
          --prompt-type macmip --type_model "$type_model" --N_exp 100  --log_dir "$dir/$dataset/$m/$val" "$extra"
        fi

        done
    done
done