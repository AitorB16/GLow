import time
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

from dataset import prepare_dataset_iid_train_common_test
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn

from flwr.client import ClientFn
from flwr.server.client_manager import ClientManager, SimpleClientManager

from custom_strategies.fedavg import FedAvg

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    #1. LOAD CONFIGURATION
    start_time = time.time()
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir + '/'
    run_id = cfg.run_name

    #LOAD TOPOLOGY
    with open(cfg.topology, 'r') as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy['num_clients']
    vcid = np.arange(num_clients) #Client IDs

    
    #2. PREAPRE YOUR DATASET
    trainloaders, validationloaders, testloaders, partitions_train, partitions_test = prepare_dataset_iid_train_common_test(num_clients, cfg.num_classes, tplgy['clients_with_no_data'], cfg.batch_size, cfg.seed)

    device = cfg.device
    
    #3. DEFINE YOUR CLIENTS
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg.num_classes, cfg.seed, device)
    cli_per_round = round(num_clients / 4)


    #4. DEFINE STRATEGY
    strategy = FedAvg(
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_fit_clients = cli_per_round,
        min_evaluate_clients = cli_per_round,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloaders),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results,
    )

    ''' Not usable currently -- change strategy or server conf during execution'''
    #strategy_pool = []
    #for cli_ID in vcid:
    #    strategy_pool.append(strategy)
    #
    #server_config_pool = []
    #for cli_ID in vcid:
    #    server_config_pool.append(fl.server.ServerConfig(num_rounds=cfg.num_rounds))
    #
    #server_pool = []
    #for cli_ID in vcid:
    #    server_pool.append(fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy))

    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)
    server = fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy)

    # Divide GPU resources among agents
    if device == 'GPU':
        num_gpus = 1.0/cli_per_round
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
    #params_path = Path(save_path) / "results.pkl"
    #results = {"history": history, "anythingelse": "here"}    
    #with open(str(params_path), "wb") as h:
    #    pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

   
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
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    out = "**losses_distributed: " + ' '.join([str(elem) for elem in history.losses_distributed]) + "\n\n**losses_avg: " + ' '.join([str(elem) for elem in history.losses_centralized])
    out = out + '\n\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']])
    out = out + '\n\n**acc_avg: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']]) + '\n'
    out = out + '\n\n**Exec_time_secs: ' + str(time.time() - start_time)
    f = open(save_path + "/raw.out", "w")
    f.write(out)
    f.close()

if __name__ == "__main__":
    main()