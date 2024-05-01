import flwr as fl
import pickle
from pathlib import Path

from typing import List, Optional, Dict

import pandas as pd
import numpy as np

import hydra
#from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import yaml

from dataset import prepare_dataset
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn

from flwr.client import ClientFn
from flwr.server.client_manager import ClientManager, SimpleClientManager


from custom_strategies.topology_based_GL import topology_based_Avg


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    #1. LOAD CONFIGURATION
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    
    #2. PREAPRE YOUR DATASET
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    #3. DEFINE YOUR CLIENTS
    vcid = np.arange(cfg.num_clients) #Client IDs
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg.num_classes)

    #LOAD TOPOLOGY - EDGES
    topology = []
    for cli_ID in vcid:
        topology.append(cfg.pools['p'+str(cli_ID)])
        #print(topology[cli_ID])


    #4. DEFINE STRATEGY
    strategy = topology_based_Avg(
        topology=topology,
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results, #LOCAL METRICS CLIENT
        total_rounds = cfg.num_rounds,
        save_path = save_path
    )

    strategy_pool = []
    for cli_ID in vcid:
        strategy_pool.append(strategy)

    server_config_pool = []
    for cli_ID in vcid:
        server_config_pool.append(fl.server.ServerConfig(num_rounds=cfg.num_rounds))

    server_pool = []
    for cli_ID in vcid:
        server_pool.append(fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy))

 
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        clients_ids = vcid,
        server = server_pool[0],
        config=server_config_pool[0],
        strategy=strategy_pool[0],
        client_resources={'num_cpus': 4, 'num_gpus': 0.2}, #num_gpus 1.0 (clients concurrently; one per GPU) // 0.25 (4 clients per GPU) -> VERY HIGH LEVEL
    )

    #6. SAVE RESULTS
    results_path = Path(save_path) / "results.pkl"

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
    f = open(save_path + "/raw_output.out", "w")
    f.write(out3)
    f.close()

if __name__ == "__main__":
    main()
