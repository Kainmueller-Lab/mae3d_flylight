#!/bin/bash

world_size=${1}
node_size=${2}
host="localhost"
host="env://"
num_cpu=${3}

job_id=$(qsub -terse -cwd -V  -l m_mem_free=2G -l h_rt=3:00:00 -l gpu=$node_size -l cuda_memory=35G -now no -pe smp ${num_cpu} -b yes python main_pretrain.py --batch_size 10 --epochs 2 --accum_iter 2 --model mae_vit_base_patch16 --input_size 96 --warmup_epochs 1 --patch_size 8 --world_size ${world_size} --num_workers 20 --num_gpu ${node_size} --dist_url $host --node_id 0)

sleep 3

echo $job_id

while true; do
    running=$(qstat -j ${job_id} -ext | grep "exec_host_list" | wc -l)
    if [[ "${running}" == 1 ]]
    then
        break
    fi
         sleep 3
done

host=$(qstat -j ${job_id} -ext | grep "exec_host_list" | tr -s "[:blank:]"  | cut -d " " -f 3 | cut -d ":" -f 1)

port=29500
host="${host}.mdc-berlin.net:${port}"

qsub -cwd -V  -l m_mem_free=2G -l h_rt=3:00:00 -l gpu=$node_size -l cuda_memory=35G -now no -pe smp ${num_cpu} -b yes python main_pretrain.py --batch_size 10 --epochs 2 --accum_iter 2 --model mae_vit_base_patch16 --input_size 96 --warmup_epochs 1 --patch_size 8 --world_size ${world_size} --num_workers 20 --num_gpu ${node_size} --dist_url $host --node_id 1
