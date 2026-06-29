#!/bin/bash

#1. PYTHON
python3 "$1" "$2" "./conf/topologies/"$4"/graph_"$5".yaml" "$3" "$5"

#2. SLURM
#sbatch exec.sbatch "$1" "$2" "./conf/topologies/"$4"/graph_"$5".yaml" "$3" "$5" #"./conf/topologies/"$3"/runtime.yaml"

#ARGVS: 1st exec name // 2nd conf_file // 3rd runtime_file // 4th root name // 5th run_id
#example:
    #./sing_exp.sh main.py conf/topologies/graph_8_2/base.yaml conf/topologies/graph_8_2/runtime.yaml graph_8_2 1

#MANUAL RUN EXAMPLE: python3 main.py conf/topologies/graph_8_2/base.yaml conf/topologies/graph_8_2/graph_0.yaml conf/topologies/graph_8_2/runtime.yaml <run_id>