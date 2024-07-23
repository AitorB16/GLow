#!/bin/bash
sbatch exec.sbatch "$1" "$2" ""$4"_""$3" "./conf/topologies/"$3"/graph_"$4".yaml"

#ARGVS: 1st exec name // 2nd conf_file // 3rd run_ID // 4th num topology
#./sing_exp.sh main.py conf/topologies/graph8_2/base.yaml graph_8_2 0
