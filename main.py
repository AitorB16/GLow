"""Argv-driven entrypoint: `python3 main.py <base.yaml>
<topology.yaml> <runtime.yaml> <run_id>` (invoked by sing/mult_exp.sh)."""

import os
import sys
import time
import pickle
import logging
from pathlib import Path

from typing import List, Optional, Dict

import pandas as pd
import numpy as np

import yaml

from datasets import prepare_dataset_iid_train_common_test, prepare_dataset_niid_train_common_test, skew_class_niid_train_common_test, skew_class_niid_train_niid_test, prepare_dataset_iid_train_iid_test, prepare_dataset_niid_train_niid_test, prepare_dataset_niid_train_iid_test
from client import cli_eval_distr_results, cli_val_distr, generate_client_fn#, weighted_average,
from server import get_on_fit_config, get_evaluate_fn

from flwr.common import Context
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.simulation import run_simulation

from custom_strategies.GLow_strategy import GLow_strategy

# silence Flower's per-round INFO logging, keep WARNING/ERROR
logging.getLogger("flwr").setLevel(logging.WARNING)


def slurm_ray_init_args():
    """Ray sizes its object store and actor pool from the whole node, not from
    the SLURM cgroup -- under sbatch it oversubscribes the allocation by orders
    of magnitude. Derive the real limits from SLURM's env vars; return {} when
    not under SLURM so local runs keep Ray's own detection."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not cpus:
        return {}

    args = {"num_cpus": int(cpus)}

    # --mem sets SLURM_MEM_PER_NODE, --mem-per-cpu sets SLURM_MEM_PER_CPU (both MB)
    mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if not mem_mb and os.environ.get("SLURM_MEM_PER_CPU"):
        mem_mb = int(os.environ["SLURM_MEM_PER_CPU"]) * int(cpus)
    if mem_mb:
        args["object_store_memory"] = int(int(mem_mb) * 1024 * 1024 * 0.15)

    scratch, job_id = os.environ.get("GLOW_RAY_TMP"), os.environ.get("SLURM_JOB_ID")
    if scratch and job_id:
        args["_temp_dir"] = f"{scratch.rstrip('/')}/{job_id}"

    return args


def main():
    # 1. LOAD CONFIGURATION AND TOPOLOGY
    start_time = time.time()

    conf_file = sys.argv[1]
    tplgy_file = sys.argv[2]
    runtime_file = sys.argv[3]
    run_id = sys.argv[4]


    with open(conf_file, 'r') as file:
        cfg = yaml.safe_load(file)

    save_path = './outputs/' + cfg['run_name'] + '/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    with open(tplgy_file, 'r') as file:
        tplgy = yaml.safe_load(file)

    num_clients = tplgy['num_clients']
    cids = np.arange(num_clients) #Client IDs

    topology = []
    for cli_ID in cids:
        topology.append(tplgy['heads']['h'+str(cli_ID)])

    with open(runtime_file, 'r') as file:
        runtime = yaml.safe_load(file)
    
    #Runtime options
    head_switch_down = []
    head_switch_up = []
    head_status = []
    head_nature = []
    head_switch_malicious = []
    for i in range(num_clients):
        client_runtime = runtime['heads']['h'+str(i)]
        head_switch_down.append(client_runtime['down'])
        head_switch_up.append(client_runtime['up'])
        head_switch_malicious.append(client_runtime['malicious'])
        head_status.append(client_runtime['status'])
        head_nature.append(client_runtime['nature'])
    
    # 2. PREAPRE YOUR DATASET
    if cfg['split_dataset'] == 'prepare_dataset_iid_train_common_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test = prepare_dataset_iid_train_common_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'prepare_dataset_niid_train_common_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test = prepare_dataset_niid_train_common_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'skew_class_niid_train_common_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test  = skew_class_niid_train_common_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'skew_class_niid_train_niid_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test  = skew_class_niid_train_niid_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'prepare_dataset_iid_train_iid_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test = prepare_dataset_iid_train_iid_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'prepare_dataset_niid_train_iid_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test = prepare_dataset_niid_train_iid_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    elif cfg['split_dataset'] == 'prepare_dataset_niid_train_niid_test':
        trainloaders, validationloaders, testloaders, class_client_matrix_train, class_client_matrix_test = prepare_dataset_niid_train_niid_test(num_clients, cfg['num_classes'], tplgy['clients_with_no_data'], cfg['batch_size'], cfg['seed'], cfg['dataset'])
    else:
        raise ValueError(
            f"Unknown split_dataset '{cfg['split_dataset']}'. Expected one of: "
            "'prepare_dataset_iid_train_common_test', 'prepare_dataset_niid_train_common_test', "
            "'skew_class_niid_train_common_test', 'skew_class_niid_train_niid_test', "
            "'prepare_dataset_iid_train_iid_test', 'prepare_dataset_niid_train_iid_test', "
            "'prepare_dataset_niid_train_niid_test'."
        )

    # 3. DEFINE YOUR CLIENTS
    client_fn = generate_client_fn(cids, trainloaders, validationloaders, cfg['num_classes'], cfg['dataset'], cfg['seed'])


    # 4. DEFINE A STRATEGY
    strategy = GLow_strategy(
        topology=topology,
        aggregation=cfg['aggregation'],
        fraction_fit=0.00001,
        fraction_evaluate=0.00001,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config(cfg['config_fit']),
        evaluate_fn=get_evaluate_fn(cfg['num_classes'], testloaders, cfg['dataset']),
        fit_metrics_aggregation_fn = cli_val_distr,
        evaluate_metrics_aggregation_fn = cli_eval_distr_results, #LOCAL METRICS CLIENT
        total_rounds = cfg['num_rounds'],
        run_id = run_id,
        num_classes=cfg['num_classes'],
        class_client_matrix_train = class_client_matrix_train,
        head_switch_down=head_switch_down,
        head_switch_up=head_switch_up,
        head_switch_malicious=head_switch_malicious,
        head_status=head_status,
        head_nature=head_nature,
        seed = cfg['seed'],
        save_path = save_path,
        warmup_rounds = cfg['warmup_rounds'],
        warmup_epochs = cfg['warmup_epochs'],
        dataset = cfg['dataset'],
    )

    server_config = ServerConfig(num_rounds=cfg['num_rounds'])

    def server_fn(context: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=strategy,
            config=server_config,
            client_manager=SimpleClientManager(),
        )

    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(client_fn=client_fn)


    # 5. RUN SIMULATIONS
    backend_config = {'client_resources': {'num_cpus': 4, 'num_gpus': 0.0}}
    ray_init_args = slurm_ray_init_args()
    if ray_init_args:
        backend_config['init_args'] = ray_init_args
        print(f" -> SLURM detected, pinning Ray to: {ray_init_args}")

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=backend_config,
    )
    # run_simulation() doesn't return a History -- strategy accumulates its own
    history = strategy.history

    # 6. SAVE RESULTS
    print('#################')
    print(str(history.losses_distributed))
    print('#################')
    print(str(history.losses_centralized))
    print('#################')
    print(str(history.metrics_distributed_fit)) #validation
    print('#################')
    print(str(history.metrics_distributed))
    print('#################')
    print(str({k: v for k, v in history.metrics_centralized.items() if k != 'preds_per_class'}))

    out = "**losses_distributed: " + ' '.join([str(elem) for elem in history.losses_distributed]) + "\n**losses_avg: " + ' '.join([str(elem) for elem in history.losses_centralized])
    out = out + '\n**acc_distr: ' + ' '.join([str(elem) for elem in history.metrics_distributed['acc_distr']]) + '\n**cid: ' + ' '.join([str(elem) for elem in history.metrics_distributed['cid']])
    out = out + '\n**acc_avg: ' + ' '.join([str(elem) for elem in history.metrics_centralized['acc_cntrl']]) + '\n**macro_f1: ' + ' '.join([str(elem) for elem in history.metrics_centralized['macro_f1']])
    out = out + '\n**Exec_time_secs: ' + str(time.time() - start_time)
    f = open(save_path + run_id + "_raw.out", "w")
    f.write(out)
    f.close()
    
    # PARTITIONS
    with open(save_path + run_id + "_partitions.out", "w") as f:
        for row in class_client_matrix_train:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("\n")
        for row in class_client_matrix_test:
            f.write(" ".join(map(str, row)) + "\n")

if __name__ == "__main__":
    main()
