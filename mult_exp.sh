#!/bin/bash
for i in $(seq 0 $4);
do
    sbatch exec.sbatch "$1" "$2" ""$i"_""$3" "./conf/topologies/"$3"/graph_"$i".yaml"
    sleep 15
    #bnd -exec python3  "$1" -m "./conf/topologies/graph_30_5/graph_"$i".yaml"
done

#ARGVS: 1st exec name // 2nd conf_file // 3rd run_ID // 4th num experiments
#./mult_exp.sh main.py conf/topologies/graph30_5/base.yaml graph_30_5 15
