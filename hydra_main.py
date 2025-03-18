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

from dataset import prepare_dataset_iid, prepare_dataset_niid
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn

from flwr.client import ClientFn
from flwr.server.client_manager import ClientManager, SimpleClientManager

from custom_strategies.topology_based_GL import topology_based_Avg


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # 1. LOAD CONFIGURATION AND TOPOLOGY
    start_time = time.time()
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir + '/'
    run_id = cfg.run_name

    with open(cfg.topology, 'r') as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy['num_clients']
    vcid = np.arange(num_clients) #Client IDs

    topology = []
    for cli_ID in vcid:
        topology.append(tplgy['pools']['p'+str(cli_ID)])

    # 2. PREAPRE YOUR DATASET
    trainloaders, validationloaders, testloader = prepare_dataset_iid(num_clients, cfg.num_classes, tplgy['clients_with_no_data'], cfg.batch_size, cfg.seed)

    device = cfg.device
    # 3. DEFINE YOUR CLIENTS
    client_fn = generate_client_fn(vcid, trainloaders, validationloaders, cfg.num_classes, device)

    # 4. DEFINE A STRATEGY
    strategy = topology_based_Avg(
        topology=topology,
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results, #LOCAL METRICS CLIENT
        total_rounds = cfg.num_rounds,
        run_id = run_id,
        early_local_train = cfg.early_local_train,
        num_classes=cfg.num_classes,
        save_path = save_path
    )

    ''' In case new strategies and configurations are deployed on run'''
    '''strategy_pool = []
    for cli_ID in vcid:
        strategy_pool.append(strategy)
    
    server_config_pool = []
    for cli_ID in vcid:
        server_config_pool.append(fl.server.ServerConfig(num_rounds=cfg.num_rounds))
    
    server_pool = []
    for cli_ID in vcid:
        server_pool.append(fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy))'''

    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)
    server = fl.server.Server(client_manager = SimpleClientManager(), strategy = strategy)

    # Divide GPU resources among agents (very high level)
    if device == 'GPU' or device == 'H100':
        num_gpus = 1.0/tplgy['max_num_clients_per_round']
    else:
        num_gpus = 0.

    # 5. RUN SIMULATIONS
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        clients_ids = vcid,
        server = server,
        config=server_config,
        strategy=strategy,
        client_resources={'num_cpus': 4, 'num_gpus': num_gpus}, #num_gpus 1.0 (clients concurrently; one per GPU) // 0.25 (4 clients per GPU) -> VERY HIGH LEVEL
    )

    # 6. SAVE RESULTS
    #results_path = Path(save_path) / "results.pkl"
    #results = {"history": history, "anythingelse": "here"}
    #with open(str(results_path), "wb") as h:
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
    out = "**losses_distributed: " + ' '.join([str(elem) for elem in history.losses_distributed]) + "\n**losses_centralized: " + ' '.join([str(elem) for elem in history.losses_centralized])
    out = out + '\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']])
    out = out + '\n**metrics_centralized: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']])
    out = out + '\n**Exec_time_secs: ' + str(time.time() - start_time)
    f = open(save_path + "/raw.out", "w")
    f.write(out)
    f.close()
    acc_distr = ''
    for i in range(cfg.num_rounds):
        acc_distr = acc_distr + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr'][i][1]])+'\n'
    f = open(save_path + "/acc_distr.out", "w")
    f.write(acc_distr)
    f.close()

if __name__ == "__main__":
    main()