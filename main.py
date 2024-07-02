import sys
import time
import flwr as fl
import pickle
from pathlib import Path

from typing import List, Optional, Dict

import pandas as pd
import numpy as np

import yaml

from dataset import prepare_dataset
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn

from flwr.client import ClientFn
from flwr.server.client_manager import ClientManager, SimpleClientManager


from custom_strategies.topology_based_GL import topology_based_Avg


def main():
    #1. LOAD CONFIGURATION

    #LOAD TOPOLOGY
    start_time = time.time()

    conf_file = sys.argv[1]
    run_id = sys.argv[2]
    tplgy_file = sys.argv[3]

    with open(conf_file, 'r') as file:
        cfg = yaml.safe_load(file)

    save_path = './outputs/' + cfg['run_name'] + '/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(tplgy_file, 'r') as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy['num_clients']
    vcid = np.arange(num_clients) #Client IDs

    topology = []
    for cli_ID in vcid:
        topology.append(tplgy['pools']['p'+str(cli_ID)])

    
    #2. PREAPRE YOUR DATASET
    trainloaders, validationloaders, testloader = prepare_dataset(num_clients, tplgy['clients_with_no_data'], tplgy['last_connected_client'], cfg['batch_size'], cfg['seed'] )

    
    device = cfg['device']
    #3. DEFINE YOUR CLIENTS
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg['num_classes'], device)


    #4. DEFINE STRATEGY
    strategy = topology_based_Avg(
        topology=topology,
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg['config_fit']),
        evaluate_fn=get_evaluate_fn(cfg['num_classes'], testloader),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results, #LOCAL METRICS CLIENT
        total_rounds = cfg['num_rounds'],
        run_id = run_id,
        early_local_train = cfg['early_local_train'],
        save_path = save_path
    )

    server_config = fl.server.ServerConfig(num_rounds=cfg['num_rounds'])
    server = fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy)

    if device == 'GPU':
        num_gpus = 1.0/tplgy['max_num_clients_per_round']
    else:
        num_gpus = 0.

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        clients_ids = vcid,
        server = server,
        config=server_config,
        strategy=strategy,
        client_resources={'num_cpus': 4, 'num_gpus': num_gpus}, #num_gpus 1.0 (clients concurrently; one per GPU) // 0.25 (4 clients per GPU) -> VERY HIGH LEVEL
    )

    #6. SAVE RESULTS
    results_path = save_path + run_id + "_results.pkl"

    results = {"history": history, "anythingelse": "here"}
    
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

   
    print('#################')
    print(str(history.losses_distributed))
    print('#################')
    print(str(history.losses_centralized))
    print('#################')
    print(str(history.metrics_distributed_fit)) #validation
    print('#################')
    print(str(history.metrics_distributed))
    print('#################')
    print(str(history.metrics_centralized))
    out1 = "**losses_distributed: " + ' '.join([str(elem) for elem in history.losses_distributed]) + "\n\n**losses_centralized: " + ' '.join([str(elem) for elem in history.losses_centralized])
    out2 = out1 + '\n\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']])
    out3 = out2 + '\n\n**metrics_centralized: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']]) + '\n'
    out4 = out3 + '\n\n**Exec_time_secs: ' + str(time.time() - start_time)
    f = open(save_path + run_id + "_raw.out", "w")
    f.write(out4)
    f.close()

if __name__ == "__main__":
    main()
