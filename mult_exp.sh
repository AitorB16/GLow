#!/bin/bash
#1st argument exec name // 2nd conf_file // 3rd run_ID // 4th tplgy file
for i in $(seq 0 $3);
do
    sbatch exec.sbatch "$1" "$2" ""$i"_30_5" "./conf/topologies/graph_30_5/graph_"$i".yaml"
    #bnd -exec python3  "$1" -m "./conf/topologies/graph_30+5/graph_"$i".yaml"
done

#./mult_exp.sh main.py conf/base.yaml 2