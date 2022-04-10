#!/bin/bash

read -p "cuda: " CUDA
read -p "models: " models
read -p "dir: " dir
read -p "prompt_k " prompt_k
read -p "datasets: " datasets
read -p "--compare_model? " trick
read -p "num_layers: " num_layers
read -p "task: " task
tasks="edgeMask contrastive attrMask VGAE"
if [ -z "$datasets" ]; then
  datasets="chameleon squirrel ACTOR TEXAS" #CORNELL AmazonComputers Citeseer Cora"
fi

declare -A dim_hidden=( ['crocodile']="64" ['squirrel']="64" ["chameleon"]="64" ['WISCONSIN']="64" ["CORNELL"]="32" ["ACTOR"]="16" ["TEXAS"]="16" ["Citeseer"]="16" ["Cora"]="16" ["AmazonComputers"]="16" )
declare -A prompt_dim_hidden=( ['crocodile']="128" ["squirrel"]="128" ['chameleon']="128" ['WISCONSIN']="64" ["CORNELL"]="64" ["ACTOR"]="64" ["TEXAS"]="64" ["Citeseer"]="256" ["Cora"]="128" ["AmazonComputers"]="256" )
#declare -A prompt_k=( ['crocodile']='5' ['squirrel']='5' ['chameleon']='5' ['WISCONSIN']="5" ["CORNELL"]="5" ["ACTOR"]="5" ["TEXAS"]="5" ["Citeseer"]="3" ["Cora"]="3" ["AmazonComputers"]="3")
declare -A alpha=( ['crocodile']='0.1' ['squirrel']='0.1' ['chameleon']='0.1' ['WISCONSIN']="0.1" ["CORNELL"]="0.1" ["ACTOR"]="0.7" ["TEXAS"]="0.7" ["Citeseer"]="0.1" ["Cora"]="0.1" ["AmazonComputers"]="0.1" )
declare -A lr=( ['crocodile']='0.2' ['squirrel']='0.02' ['chameleon']='0.02' ['WISCONSIN']="0.02" ["CORNELL"]="0.02" ["ACTOR"]="0.02" ["TEXAS"]="0.02" ["Citeseer"]="0.006" ["Cora"]="0.006" ["AmazonComputers"]="0.006" )
declare -A cont=( ['crocodile']='False' ['squirrel']='False' ['chameleon']='False' ['WISCONSIN']="False" ["CORNELL"]="False" ["ACTOR"]="False" ["TEXAS"]="False" ["Citeseer"]="False" ["Cora"]="True" ["AmazonComputers"]="False" )


w_org="1"
if [ $prompt_k = '0' ]; then
  echo "no org features"
  w_org="0"
fi

for val in $tasks; do
  type_model="GCN"
  if [ "$val" == "VGAE" ]; then
    type_model="VGAE"
  fi
  for m in $models; do
    for dataset in $datasets; do
        layers="$num_layers"
        if [ "$m" == "GCNII" ] ; then
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
        --prompt-k "$prompt_k" \
        --prompt-layer 3 \
        --prompt-lr 0.2 \
        --prompt-pretrain-lr 0.03 \
        --prompt-pretrain-type "$val" \
        --prompt-temp 2.0\
        --prompt-type macmip \
        --type_model "$type_model" \
        --N_exp 100  \
        --log_dir "$dir/$dataset/$m/$val" \
        --prompt-continual "${cont[$dataset]}" \
        --downstream-task "$task" \
        --prompt-w-org-features "$w_org"

        done
    done
done