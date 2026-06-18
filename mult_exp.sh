#!/bin/bash
for i in $(seq 0 $4);
do
    #1. PYTHON
    python3 "$1" "$2" "./conf/topologies/"$3"/graph_"$i".yaml" "$3" "$i" #"./conf/topologies/"$3"/runtime.yaml"
    
    #2. SLURM
    #sbatch exec.sbatch "$1" "$2" "./conf/topologies/"$4"/graph_"$i".yaml" "$3" "$i" #"./conf/topologies/"$3"/runtime.yaml"
    #sleep 30
done

#ARGVS: 1st exec name // 2nd conf_file // 3rd runtime_file // 4th root name
#example:
    #./custom_exp.sh main.py conf/topologies/graph_8_2/base.yaml conf/topologies/graph_8_2/runtime.yaml graph_8_2

#MANUAL RUN EXAMPLE: python3 main.py conf/topologies/graph_8_2/base.yaml conf/topologies/graph_8_2/graph_0.yaml conf/topologies/graph_8_2/runtime.yaml <run_id>