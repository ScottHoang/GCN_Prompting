#!/bin/bash

read -p "cuda: " CUDA
read -p "models: " models
read -p "dir: " dir
read -p "prompt_k " prompt_k
read -p "datasets: " datasets
read -p "--compare_model? " trick
read -p "num_layers: " num_layers
read -p "task: " task
read -p "aggr: " aggr
read -p "N_exp " N_exp

if [ -z "$N_exp" ]; then
  N_exp="100"
fi

if [ -z "$aggr" ]; then
  aggr='edges'
fi
tasks="edgeMask VGAE"  #contrastive attrMask"

if [ -z "$datasets" ]; then
  datasets="squirrel chameleon ACTOR TEXAS WISCONSIN CORNELL"
fi

declare -A dim_hidden=( ['crocodile']="64" ['squirrel']="64" ["chameleon"]="64" ['WISCONSIN']="64" ["CORNELL"]="32" ["ACTOR"]="16" ["TEXAS"]="16" ["Citeseer"]="64" ["Cora"]="64" ["AmazonComputers"]="64" )
declare -A prompt_dim_hidden=( ['crocodile']="128" ["squirrel"]="128" ['chameleon']="128" ['WISCONSIN']="64" ["CORNELL"]="64" ["ACTOR"]="64" ["TEXAS"]="64" ["Citeseer"]="64" ["Cora"]="64" ["AmazonComputers"]="256" )
declare -A alpha=( ['crocodile']='0.1' ['squirrel']='0.1' ['chameleon']='0.1' ['WISCONSIN']="0.1"
["CORNELL"]="0.1" ["ACTOR"]="0.1" ["TEXAS"]="0.1" ["Citeseer"]="0.1" ["Cora"]="0.1" ["AmazonComputers"]="0.1" )
declare -A lr=( ['crocodile']='0.2' ['squirrel']='0.02' ['chameleon']='0.02' ['WISCONSIN']="0.02" ["CORNELL"]="0.02" ["ACTOR"]="0.02" ["TEXAS"]="0.02" ["Citeseer"]="0.02" ["Cora"]="0.02" ["AmazonComputers"]="0.02" )
declare -A cont=( ['crocodile']='0' ['squirrel']='0' ['chameleon']='0' ['WISCONSIN']="0" ["CORNELL"]="0" ["ACTOR"]="0" ["TEXAS"]="0" ["Citeseer"]="0" ["Cora"]="0" ["AmazonComputers"]="0" )


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
        --embedding_dropout 0.8 \
        --epochs "$epochs"\
        --lr "${lr[$dataset]}" \
        --num_layers "$layers" \
        --prompt-aggr "$aggr" \
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
        --N_exp "$N_exp"  \
        --log_dir "$dir/$dataset/$m/$val" \
        --prompt-continual "${cont[$dataset]}" \
        --downstream-task "$task" \
        --prompt-w-org-features "$w_org"

        done
    done
done
