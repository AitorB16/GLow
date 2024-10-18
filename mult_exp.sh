#!/bin/bash
for i in $(seq 0 $4);
do
    #1. PYTHON
    #python3 "$1" "$2" ""$i"_""$3" "./conf/topologies/"$3"/graph_"$i".yaml"
    
    #2. SLURM
    sbatch exec.sbatch "$1" "$2" ""$i"_""$3" "./conf/topologies/"$3"/graph_"$i".yaml"
    #sleep 30
done

#ARGVS: 1st exec name // 2nd conf_file // 3rd run_ID // 4th num experiments
#example:
    #./mult_exp.sh main.py conf/topologies/graph_8_2/base.yaml graph_8_2 4
