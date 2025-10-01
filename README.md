# GLow - A Flower Based GL Strategy
GLow is a novel Gossip Learning (GL) strategy for simulating fully distributed systems using the **Flower Framework**. The implementation is able to simulate a fully decentraliced network composed by virtual network agents (disposed in different toplogies) that perform parameter aggregation with their neighbors. Modularity is an essential part of the system making the usage of different datasets, models and run configurations easy to integrate. Although the [Flower Framework](https://flower.ai/docs/framework/how-to-implement-strategies.html) guidelines for strategy implementation are followed, GLow differs from a vanilla FL scheme -- there is no aggregation server and each agent operates as client and server at the same time (P2P). Further explanations of the strategy are found in [GLow - A Novel, Flower-Based Simulated Gossip Learning Strategy](https://arxiv.org/abs/2501.10463) manuscript.

Moreover, a centralized (CNL) and a vanilla Federated (FL) version of the system are provided as well; in order to have a wider testbench and give researchers a robust comparison baseline -- they are completely integrated with parts of GLow implementation and configuration.

## Directory structure
- **conf (configuration files)**: YAML configuration files
- **conf/topologies (topologies architecture)**: YAML files describing system topology
- **custom_strategies**: Custom GLow strategy with vanilla FedAVG
- **flwr_lib_modifications**: Updated files from internal Flower architecture
- **visualization**: Notebooks to visualize GLow and vanilla FL outputs as well as topology architecture (graphs)


## Dependencies
Install the required libraries present on [deps.req](deps.req).

Built upon Flower v1.7.0.
> Note: Python3.10 and pip3 package installer are recommended.

## Dataset
Download or create a custom dataset, the implementation is currently designed to work with [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) which should be downloaded, extracted and placed into the [dataset](../dataset) directory.

> Note: Other branches working with MNIST are available in the repository.
## Configuration files

Configuration files in GLow are composed by a base file and a topology file.
> Note: The system is designed to parse YAML files.

### Base

Example file located in [conf/base.yaml](conf/base.yaml) following structure:
- **run_name:** str; run name
- **aggregation:** str; aggragation algorithm, select among *'inplace'*, *'score'*, *'score_neigh_params'*
- **topology:** str; path to yaml file containing system topology
- **split_dataset:** str; split dataset among agents, select among *'prepare_dataset_iid_train_common_test'*, *'prepare_dataset_niid_train_common_test'*,*'prepare_dataset_iid_train_iid_test'*, *'prepare_dataset_niid_train_iid_test'*, *'prepare_dataset_niid_train_niid_test'*, *'prepare_dataset_niid_class_partition'*
- **device:** str; select among *CPU*, *GPU*, *H100*
- **early_local_train:** bool; to force the system work in SL for the first *n* communication rounds before neighbor aggregation 
- **num_rounds:** int; total number of communication rounds
- **batch_size:** int; hyperparameter for DataLoader
- **num_classes:** int; output layer size
- **seed:** int, added for replicability
- **config_fit:** hyperparameters
    - **lr:** float; learning rate
    - **momentum:** float; momentum
    - **local_epochs:**  int; epochs to be performed by each agent with local instances

### Topology

Example file located in [conf/topologies/graph_8_2/graph_1.yaml](conf/topologies/graph_8_2/graph_1.yaml) following structure:
- **num_clients:** int; total number of agents
- **max_num_clients_per_round:** int; max number of clients performing aggregation (i.e., number of neighbors)
- **clients_with_no_data:** int list (optional); containing the IDs of special nodes with no local instances
- **last_connected_client:** int; ID of last node connected to the network, nodes with higher IDs will perform SL
- **pools:**
    - **p0:** int list; containing neighbor IDs
    - **p1:** int list; containing neighbor IDs
    - **...** 
    - **p<num_clients-1>:** int list; containing neighbor IDs

> Note: Multiple example topologies are located in [topologies](./conf/topologies/) directory; chain, ring_chain, ring, star_chain, graph_8_2 (8+2 from disconnected to fully connected) and graph_16_2 (16+4 from disconnected to fully connected) cases.
## Topology generator
[generate_topology.ipynb](./generate_topology.ipynb) Allows to easily create YAML files for different topologies. All methods need a positive int specifying the number of inter-connected nodes.
- **generate_chain():**
- **generate_star_chain():**
- **...**
- **generate_from_islands_to_fully_connected():** this method creates itertively all the possible topologies from a disconnected graph to a fully connected one (connecting each node to 2 new neighbors in each iteration)

> Note: All methods admit an optional parameter (int) specifying the number of islands (disconnected nodes).
## Execution instructions

The following execution variants are allowed with their corresponding HPC deployment scripts -- depending on the execution target: (1) Single run unique topology. (2) Multiple simulations multiple topologies.

### Single run (without arguments) -- Recommended
[Hydra Framework](https://hydra.cc/) is used to allow researchers track simulation outputs easily. It is recommended for executions involving custom topologies -- an specific graph. 

```sh 
python3 hydra_main.py
```
A directory containing current date, time and all execution outputs is created by Hydra.
> Note: line '@hydra.main(config_path="conf", config_name="base", version_base=None)' in *hydra_main.py* specifies configuration file location.

#### FL version

Integration under Flower allows to easily switch among strategies; a vanilla FedAVG system working under the same configuration files is provided.

```sh 
python3 FL_hydra_main.py
```

### Multiple runs (with arguments)

For experiments that involve multiple graphs; e.g., adding edges to an specific number of agents. Deployment is thought to be done using [mult_exp.sh](./mult_exp.sh) script, with topologies generated by [generate_topology.ipynb](./generate_topology.ipynb) and placed in [conf/topologies](./conf/topologies/) directory. Example [graph_8_2](./conf/topologies/graph_8_2/), contains 5 topologies with the same number of nodes and configurations, going from fully disconnected to fully connected graphs.

```sh
sh mult_exp.sh main.py <conf_file.yaml> <run_name> <total number of topologies - 1>
```

A directory named *run_name* is created in [outputs](./outputs/) to store execution results.
> Note: Deployment script [mult_exp.sh](/mult_exp.sh) has to be configured for Python or Slurm usage.

### Single runs (with arguments)

Following the same structure of [mult_exp.sh](./mult_exp.sh), script [sing_run.sh](./sing_exp.sh) is provided for launching an specific inter-connectied graph generated by [generate_topology.ipynb](./generate_topology.ipynb).

```sh
sh sing_exp.sh main.py <conf_file.yaml> <run_name> <specific topology ID>
```
> Note: Topology ID is an integer corresponding to the interconnection degree.

## Results
The output of each experiment consists in the following files:

- **pool.out:** Accuracies and Losses obtained by each pool (node head) after *n* communication rounds
- **raw.out:** Full output; *losses_distributed*, *losses_avg, *acc_distr*, *cid*, *acc_avg*, *Exec_time*
- **parameters/:** Directory containing torch parameters per agent after *n* communication rounds; *<agent_id>.pth*

> Note: **acc_distr.out:** Contains distributed accuracies obtained by each agent per communication round -- parsed version of raw.out for visualization purposes.

### Executions with Hydra
Hydra creates a nested directory with current date and times at the moment experiments are launched. Additional files and directories are created in this execution variant

- **hydra_main.log:** Full system log for debbuging
- **.hydra/**
    - **config.yaml:** A copy of the configuration file specified in *@hydra.main(...)*
    - **...**

## Visualization

This section is oriented for visualization of the results obtained from *multiple runs*; experiments testing system convergence by the degree of inter-connectivity -- from fully disconnected to fully connected. Moreover, a version for visualizing the results from FedAVG is available as well.

- **8_visualize_results:** Visualize GLow experiments for an 8 agent scenario
- **16_visualize_results:** Visualize GLow experiments for an 16 agent scenario
- **FL_visualize_results:** Visualize FL experiments
- **draw_graphs:** Draws graphs from YAML files generated with [generate_topology.ipynb](./generate_topology.ipynb)

## Changes in libraries

Dealing with control nodes with no local data is not a Flower feature. Performing weighted average among network parameters triggers scaling factors realted issues (division by 0). Hence, *aggregate_inplace()*  method of [flwr/server/strategy/aggregate.py](/flwr_lib_modifications/aggregate.py) is modified. Scaling factor for nodes with no local data is set to 1.0 -- no scaling factor applied. 

## Author

* **Aitor Belenguer** 

## License

MIT
