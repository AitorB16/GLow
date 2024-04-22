import flwr as fl
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import yaml

from dataset import prepare_dataset
from client import cli_eval_distr_results, cli_val_distr, meta_generate_client_fn#, generate_client_fn, weighted_average, 
from server import get_on_fit_config, get_evaluate_fn


from custom_strategies.cli_map_FedAvg import cli_FedAvg

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    #1. LOAD CONFIGURATION
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    
    #2. PREAPRE YOUR DATASET
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    #3. DEFINE YOUR CLIENTS
    #client_fn = generate_client_fn(trainloaders, validationloaders, first_layer_size, cfg.num_classes)
    vcid = np.arange(cfg.num_clients) #Client IDs
    client_fn = meta_generate_client_fn(vcid, trainloaders, validationloaders, cfg.num_classes)

    #4. DEFINE STRATEGY
    #strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
    #                                     min_fit_clients=cfg.num_clients_per_round_fit,
    #                                     fraction_evaluate=0.00001,
    #                                     min_evaluate_clients=cfg.num_clients_per_round_eval,
    #                                     min_available_clients=cfg.num_clients,
    #                                     on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    #                                     evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    #                                     fit_metrics_aggregation_fn = cli_val_distr,
    #                                     evaluate_metrics_aggregation_fn = cli_eval_distr_results #LOCAL METRICS CLIENT
    #                                     )
    
    strategy = cli_FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,
                                         fraction_evaluate=0.00001,
                                         min_evaluate_clients=cfg.num_clients_per_round_eval,
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
                                         fit_metrics_aggregation_fn = cli_val_distr,
                                         evaluate_metrics_aggregation_fn = cli_eval_distr_results #LOCAL METRICS CLIENT
                                         )


    #5. SIMULATE
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 4, 'num_gpus': 0.25}, #num_gpus 1.0 (clients concurrently; one per GPU) // 0.25 (4 clients per GPU) -> VERY HIGH LEVEL
    )

    #6. SAVE RESULTS
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}
    
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

   
    #out = repr(history)
    #print(out)
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
    out2 = out1 + '\n\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']]) + '\n\n** cli_sample: ' +  ' '.join([str(elem) for elem in history.metrics_distributed['idx']])
    out3 = out2 + '\n\n**metrics_centralized: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']]) + '\n'
    f = open(save_path + "/output.out", "w")
    f.write(out3)
    f.close()
    #out = {'Accuracy': [accuracy], 'Training_Loss': train_loss}
    #out = pd.DataFrame(list(out.items()), columns=['Acc', 'Training_Loss'])
    #out.to_csv(save_path + "/output.csv", sep='\t', encoding='utf-8')


if __name__ == "__main__":
    main()
