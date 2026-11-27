# Copyright 2020 Flower Labs GmbH. Modified for GLow, by A. Belenguer 2027
# SPDX-License-Identifier: Apache-2.0
"""
Rotating head: each round one node is head, trains, and aggregates its up
neighbours' models into its own. A 'down' head wastes its round, an 'sl' head
trains alone; down/sl nodes take no part in other heads' rounds.
"""

import os
import time
import json
import flwr
import numpy as np
from collections import OrderedDict
import torch
from models import build_model, pack_ndarrays

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History

from flwr_lib_modifications.aggregate import self_learning, aggregate_inplace, aggregate_score, aggregate_score_validation, aggregate_score_centroids_2, aggregate_score_centroids_2b, aggregate_trimmed_mean, aggregate_krum
from flwr.server.strategy.strategy import Strategy

from  flwr.server.criterion import Criterion

from flwr.common.typing import GetParametersIns, GetPropertiesIns


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class GLow_vanilla(Strategy):
    """Decentralized gossip-learning strategy. https://www.sciencedirect.com/science/article/pii/S074373152600050X
    See module docstring for the rotating-head mechanic.
    """

    name = 'vanilla'

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        total_rounds: int,
        aggregation: str,
        topology: List[List[int]],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2, # overwritten per-round from topology
        min_evaluate_clients: int = 2, # overwritten per-round from topology
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[List[Parameters]] = None,
        head_parameters: Optional[List[Parameters]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        run_id: str,
        num_classes: int,
        class_client_matrix_train: List[List[int]],
        head_switch_down: List[List[int]],
        head_switch_up: List[List[int]],
        head_switch_malicious: List[List[int]],
        head_switch_random: List[List[int]],
        head_switch_sl: List[List[int]],
        head_status: List[str],
        head_nature: List[str],
        seed: int,
        save_path: str,
        warmup_rounds: Optional[int] = None,
        warmup_epochs: int = 15,
        dataset: str = 'mnist',
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.total_rounds = total_rounds
        self.current_round = 0
        self.cur_glob_round = 0
        self.topology = topology
        self.aggregation = aggregation
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.client_list = np.arange(min_available_clients).tolist()
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.head_parameters = head_parameters
        self.selected_head = None
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.head_metrics = [None] * self.min_available_clients
        self.head_losses = [None] * self.min_available_clients
        self.head_f1 = [None] * self.min_available_clients
        self.head_preds_per_class = [np.zeros((num_classes, num_classes), dtype=int) for _ in range(self.min_available_clients)]
        self.run_id = run_id
        self.training_time = [0.] * self.min_available_clients
        self.aggregation_time = [0.] * self.min_available_clients
        self.num_classes = num_classes
        self.class_client_matrix_train = class_client_matrix_train
        self.seed = seed
        self.save_path = save_path
        self.warmup_rounds = warmup_rounds
        self.warmup_epochs = warmup_epochs
        self.out_buf = ''
        self.dataset = dataset
        self.head_switch_down = head_switch_down
        self.head_switch_up = head_switch_up
        self.head_switch_malicious = head_switch_malicious
        self.head_switch_random = head_switch_random
        self.head_switch_sl = head_switch_sl
        self.head_status = head_status
        self.head_nature = head_nature
        # ClientProxy.cid -> topology index; simulation ID maps to client ID
        self._cid_to_index: Dict[str, int] = {}
        self.history = History()
        # Caches aggregate_fit()'s per-neighbour eval for neighbours
        self._last_eval_results = {}
        # For raw output
        self._last_central = {}
        self._last_distr = {}
        self.round_log = []
        
        
        # per-node list of neighbour accuracies/params
        self.neigh_metrics = []
        for i in range(min_available_clients):
            self.neigh_metrics.append([])
            for j in range(min_available_clients):
                self.neigh_metrics[i].append(None)

    def _round_seed(self, server_round: int, node_index: int) -> int:
        # independent of client scheduling order (Ray reuses workers)
        return self.seed + server_round * 10_000 + node_index

    def _cids_for_indices(self, indices) -> List[str]:
        index_set = set(indices)
        return [cid for cid, idx in self._cid_to_index.items() if idx in index_set]

    def get_up_neighbors(self):
        """Up neighbours of the head (itself included); [head] if 'sl', [] if 'down'."""
        status = self.head_status[self.selected_head]
        if status == 'down':
            return []
        if status == 'sl':
            return [self.selected_head]
        return [n for n in self.topology[self.selected_head] if self.head_status[n] == 'up']

    def head_check(self):
        """Apply this round's scheduled up/down/malicious/random transitions."""
        for agent in self.client_list:
            if self.head_switch_up[agent] is not None:
                if self.current_round in self.head_switch_up[agent]:
                    self.head_status[agent] = 'up'
            if self.head_switch_down[agent] is not None:
                if self.current_round in self.head_switch_down[agent]:
                    self.head_status[agent] = 'down'
            if self.head_switch_malicious[agent] is not None:
                if self.current_round in self.head_switch_malicious[agent]:
                    self.turn_bad(agent, 'malicious')
            if self.head_switch_random[agent] is not None:
                if self.current_round in self.head_switch_random[agent]:
                    self.turn_bad(agent, 'random')
            if self.head_switch_sl[agent] is not None:
                if self.current_round in self.head_switch_sl[agent]:
                    self.head_status[agent] = 'sl'

    def turn_bad(self, agent, nature):
        """Make `agent` a 'malicious' or 'random' node (label noise in
        models.corrupt_labels): isolate it and restart it from scratch."""
        self.head_nature[agent] = nature
        self.topology[agent] = [agent]
        self.head_metrics[agent] = None
        self.head_losses[agent] = None
        self.head_f1[agent] = None
        self.head_preds_per_class[agent] = np.zeros((self.num_classes, self.num_classes),dtype=int)
        self.neigh_metrics[agent] = [None] * self.min_available_clients
        self.head_parameters[agent] = self.initial_parameters[agent]

    def select_head(self):
        """Next node in client_list becomes head (even if 'down'); apply the schedule."""
        self.selected_head = self.client_list[0]
        self.client_list = np.roll(self.client_list, -1).tolist()
        self.current_round += 1
        self.head_check()
        if self.current_round % self.min_available_clients == 0:
            self.cur_glob_round += 1

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        self.min_fit_clients = len(self.topology[self.selected_head])
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        self.min_evaluate_clients = len(self.topology[self.selected_head])
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Map cids to topology indices, store initial parameters, pick the first head."""
        clients = client_manager.sample(self.min_available_clients)

        props_ins = GetPropertiesIns(config={})
        for client in clients:
            props_res = client.get_properties(ins=props_ins, timeout=None, group_id=0)
            self._cid_to_index[client.cid] = int(props_res.properties["partition_id"])

        ins = GetParametersIns(config={})

        if self.initial_parameters is None:
            self.initial_parameters = [None] * self.min_available_clients
            self.head_parameters = [None] * self.min_available_clients

            for client in clients:
                index = self._cid_to_index[client.cid]
                self.initial_parameters[index] = client.get_parameters(ins=ins, timeout=None, group_id=0).parameters
                self.head_parameters[index] = self.initial_parameters[index]

        self.current_round = -1
        self.select_head()
        self.client_list = np.roll(self.client_list, 1).tolist()
        initial_parameters = self.initial_parameters[self.selected_head]
        self.current_round = -1
        self.cur_glob_round = -1
        return initial_parameters

    def save_results(self, server_round):
        """Per round: traces line, console line and the head's parameters."""
        if self.current_round < 0: 
            return
        # neighbours: the nodes that took part (just the head when down/sl);
        # a down head's line repeats its last metrics (its model didn't change)
        shown = self.get_up_neighbors() or [self.selected_head]
        self.out_buf += f'head_ID: {str(self.selected_head)} round ID: {self.cur_glob_round} neighbours: {str(shown)} loss_test: {str(self.head_losses[self.selected_head])} acc_test: {str(self.head_metrics[self.selected_head])} f1_test: {str(self.head_f1[self.selected_head])} training_time: {str(self.training_time[self.selected_head])} status: {self.head_status[self.selected_head]}\n'
        head = self.selected_head
        num = lambda v: '-' if v is None else f'{v:.3f}'
        print(f'round {server_round}/{self.total_rounds}  glob {self.cur_glob_round}  head {head}  '
              f'loss {num(self.head_losses[head])}  acc {num(self.head_metrics[head])}  '
              f'f1 {num(self.head_f1[head])}  train {self.training_time[head]:.1f}s  aggregation {self.aggregation_time[head]:.3f}ms', flush=True)


        param_path = f'{self.save_path}{self.run_id}_parameters/'
        os.makedirs(param_path, exist_ok=True)
        net = build_model(self.dataset, self.num_classes)
        cli_params_ndarrays = parameters_to_ndarrays(self.head_parameters[self.selected_head])
        params_dict = zip(net.state_dict().keys(), cli_params_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        #Save head params per round
        torch.save(net.state_dict(), f'{param_path}r{server_round}_h{self.selected_head}.pth')
        if server_round == self.total_rounds:
            self.save_on_finish()

    def save_on_finish(self):
        """Write _traces.out, _heads.out, _result_matrix.out and final parameters."""
        #Save mean of training and aggregation time as well? -- Will need a var to increment and divide by number of rounds
        f = open(f"{self.save_path}{self.run_id}_traces.out", "w")
        f.write(self.out_buf)
        f.close()
        num = lambda v: '-' if v is None else f'{v:.3f}'
        out = ''
        for cli_ID in range(self.min_available_clients):
            out = out + 'head_ID: ' + str(cli_ID) + ' neighbours: ' + str(self.topology[cli_ID]) + f' loss_test: {num(self.head_losses[cli_ID])}' + f' acc_test: {num(self.head_metrics[cli_ID])}' + f' f1_test: {num(self.head_f1[cli_ID])} \n'
        f = open(self.save_path + str(self.run_id) + "_heads.out", "w")
        f.write(out)
        f.close()

        with open(self.save_path + str(self.run_id) + "_result_matrix.out", "w") as f:
            for cli_ID in range(self.min_available_clients):
                for row in self.head_preds_per_class[cli_ID]:
                    f.write(" ".join(map(str, row)) + "\n")
                f.write("\n")

        param_path = self.save_path + str(self.run_id) + '_parameters/'
        os.makedirs(param_path, exist_ok=True)
        for cli_ID in range(self.min_available_clients):
            net = build_model(self.dataset, self.num_classes)
            cli_params_ndarrays = parameters_to_ndarrays(self.head_parameters[cli_ID])
            params_dict = zip(net.state_dict().keys(), cli_params_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), param_path + str(cli_ID) + '.pth')

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Re-evaluate this round's up-neighbours; return the head's own
        (loss, metrics)."""
        if self.evaluate_fn is None:
            return None

        if server_round > 0:
            head, up = self.selected_head, self.get_up_neighbors()
            self.round_log.append((server_round, head, [head] * (head in up) + sorted(n for n in up if n != head)))

        if self.head_status[self.selected_head] == 'down':  # wasted round: repeat the last values
            head = self.selected_head
            if head in self._last_central:
                loss, metrics = self._last_central[head]
                self.history.add_loss_centralized(server_round=server_round, loss=loss)
                self.history.add_metrics_centralized(server_round=server_round, metrics=metrics)
            if head in self._last_distr:  # the head's own last loss; nobody aggregated
                self.history.add_loss_distributed(server_round=server_round, loss=self._last_distr[head])
                self.history.add_metrics_distributed(server_round=server_round, metrics={'acc_val': [], 'cid': []})
            self.save_results(server_round)
            return None

        head_loss = None
        head_metrics = None

        up_neighbours = self.get_up_neighbors()
        for i, neighbour in enumerate(up_neighbours):
            # Just recompute eval in head, whose parameters just changed.
            if neighbour != self.selected_head and neighbour in self._last_eval_results:
                loss, metrics = self._last_eval_results[neighbour]
            else:
                parameters_ndarrays = parameters_to_ndarrays(self.head_parameters[neighbour])
                config = {'nature': self.head_nature[self.selected_head], 'seed': self._round_seed(server_round, neighbour)}
                eval_res = self.evaluate_fn(self.selected_head, server_round, parameters_ndarrays, config)

                if eval_res is None:
                    return None

                loss, metrics = eval_res
                self._last_eval_results[neighbour] = (loss, metrics)

            self.neigh_metrics[self.selected_head][neighbour] = metrics['acc_test']

            if neighbour == self.selected_head:
                self.head_losses[self.selected_head] = loss
                self.head_metrics[self.selected_head] = metrics['acc_test']
                self.head_f1[self.selected_head] = metrics['macro_f1_test']
                self.head_preds_per_class[self.selected_head] = metrics['preds_per_class']
                head_loss = loss
                head_metrics = metrics

        if head_loss is None:
            raise RuntimeError(
                f"Head {self.selected_head} is missing from topology[{self.selected_head}]; "
                "every row must list its own node."
            )

        self.save_results(server_round)

        self.history.add_loss_centralized(server_round=server_round, loss=head_loss)
        self.history.add_metrics_centralized(server_round=server_round, metrics=head_metrics)
        self._last_central[self.selected_head] = (head_loss, head_metrics)

        return head_loss, head_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Rotate head, build one FitIns per up-neighbour from the head's
        parameters."""
        
        self.select_head()
        connections = self.get_up_neighbors()
        if not connections:  # 'down' head: the round is wasted
            return []

        class select_criterion(Criterion):
            def __init__(self, cid_list):
                self.cid_list = cid_list
            def select(self, client: ClientProxy) -> bool:
                return client.cid in self.cid_list

        clients = client_manager.sample(
            num_clients=len(connections), criterion=select_criterion(self._cids_for_indices(connections))
        )
        if self.aggregation == 'approach_2b':
            # neighbours compute dist_N against the head's current (pre-training) params
            head_params = pack_ndarrays(parameters_to_ndarrays(self.head_parameters[self.selected_head]))
        pairs = []
        for client in clients:
            index = self._cid_to_index[client.cid]
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            if self.aggregation == 'approach_2b' and index != self.selected_head:
                config['head_params'] = head_params
            config['strategy'] = self.name
            config['aggregation'] = self.aggregation
            # JSON: configs only take scalars
            config['neighbors'] = json.dumps(connections)
            config['head_cid'] = self.selected_head
            config['comm_round'] = server_round
            config['warmup_rounds'] = self.warmup_rounds
            config['warmup_epochs'] = self.warmup_epochs
            config['nature'] = self.head_nature[self.selected_head]
            config['seed'] = self._round_seed(server_round, index)

            fit_ins = FitIns(self.head_parameters[index], config)
            pairs.append((client, fit_ins))
        return pairs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Per up-neighbour EvaluateIns for its own current
        (pre-aggregation) parameters."""

        class select_criterion(Criterion):
            def __init__(self, cid_list):
                self.cid_list = cid_list
            def select(self, client: ClientProxy) -> bool:
                return client.cid in self.cid_list

        if self.fraction_evaluate == 0.0:
            return []

        connections = self.get_up_neighbors()
        if not connections:  # 'down' head
            return []

        clients = client_manager.sample(
            num_clients=len(connections), criterion=select_criterion(self._cids_for_indices(connections))
        )

        pairs = []
        for client in clients:
            index = self._cid_to_index[client.cid]
            config = {}
            if self.on_evaluate_config_fn is not None:
                config = self.on_evaluate_config_fn(server_round)
            config['head_cid'] = self.selected_head
            config['nature'] = self.head_nature[self.selected_head]
            config['seed'] = self._round_seed(server_round, index)

            evaluate_ins = EvaluateIns(self.head_parameters[index], config)
            pairs.append((client, evaluate_ins))
        return pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Fold up-neighbours' fit results into the head's new parameters
        via self.aggregation (flwr_lib_modifications/aggregate.py)."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        # results come in completion order, which Ray varies; sort for reproducibility
        results = sorted(results, key=lambda r: self._cid_to_index[r[0].cid])

        # Re-evaluate each up-neighbour's pre-aggregation parameters to
        # refresh neigh_metrics
        up_neighbours = self.get_up_neighbors()
        for i, neighbour in enumerate(up_neighbours):
            parameters_ndarrays = parameters_to_ndarrays(self.head_parameters[neighbour])
            config = {'nature': self.head_nature[self.selected_head], 'seed': self._round_seed(server_round, neighbour)}
            eval_res = self.evaluate_fn(self.selected_head, server_round, parameters_ndarrays, config)

            if eval_res is None:
                return None, {}
            loss, metrics = eval_res
            self._last_eval_results[neighbour] = (loss, metrics)
            self.neigh_metrics[self.selected_head][neighbour] = metrics['acc_test']

            if neighbour == self.selected_head:
                self.head_losses[self.selected_head] = loss
                self.head_metrics[self.selected_head] = metrics['acc_test']
                self.head_f1[self.selected_head] = metrics['macro_f1_test']
                self.head_preds_per_class[self.selected_head] = metrics['preds_per_class']

        # ClientProxy.cid is a simulation id, needs to be mapped to client ID
        results_by_index = {
            self._cid_to_index[cli.cid]: fit_res for cli, fit_res in results
        }
        #Measure time
        start_event = time.perf_counter()
        if self.current_round <= self.warmup_rounds or self.head_status[self.selected_head] == 'sl':
            aggregated_ndarrays = self_learning(results_by_index,up_neighbours, self.selected_head)
        elif self.aggregation == 'inplace':
            aggregated_ndarrays = aggregate_inplace(results_by_index)
        elif self.aggregation == 'krum':
            aggregated_ndarrays = aggregate_krum(results_by_index, 1, 2)
        elif self.aggregation == 'trimmed_mean':
            aggregated_ndarrays = aggregate_trimmed_mean(results_by_index, 1)
        elif self.aggregation == 'score':
            aggregated_ndarrays = aggregate_score(results_by_index, self.neigh_metrics[self.selected_head], up_neighbours, self.selected_head)
        elif self.aggregation == 'score_validation':
            aggregated_ndarrays = aggregate_score_validation(results_by_index, up_neighbours, self.selected_head)
        elif self.aggregation == 'approach_2':
            aggregated_ndarrays = aggregate_score_centroids_2(results_by_index, up_neighbours, self.selected_head, self.current_round, self.class_client_matrix_train, self.num_classes, 0.4, 0.6, 0.)
        elif self.aggregation == 'approach_2b':
            aggregated_ndarrays = aggregate_score_centroids_2b(results_by_index, up_neighbours, self.selected_head, self.class_client_matrix_train, self.num_classes)
        else:
            raise ValueError(
                f"Unknown aggregation strategy '{self.aggregation}'. Expected one of: "
                "'inplace', 'krum', 'trimmed_mean', 'score', 'score_validation', 'approach_2', 'approach_2b'."
            )
        
        end_event = time.perf_counter()
        aggregation_time = end_event - start_event

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Only the head's own parameters are updated
        self.head_parameters[self.selected_head] = parameters_aggregated

        self.training_time[self.selected_head] = metrics_aggregated['training_time'][metrics_aggregated['cid'].index(self.selected_head)]
        self.aggregation_time[self.selected_head] = aggregation_time * 1000

        self.history.add_metrics_distributed_fit(server_round=server_round, metrics=metrics_aggregated)

        return parameters_aggregated, metrics_aggregated

    def _val_loss(self, results) -> float:
        """losses_val: the head's own loss (only its model changes in a round)."""
        return next(res.loss for proxy, res in results if self._cid_to_index[proxy.cid] == self.selected_head)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Collect configure_evaluate()'s per-neighbour loss/metrics."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # see aggregate_fit: completion order is not reproducible
        results = sorted(results, key=lambda r: self._cid_to_index[r[0].cid])

        loss_aggregated = self._val_loss(results)
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        if 'cid' in metrics_aggregated:  # head first, then ascending: the order of round_log
            pos = sorted(range(len(metrics_aggregated['cid'])), key=lambda i: metrics_aggregated['cid'][i] != self.selected_head)
            metrics_aggregated = {k: [v[i] for i in pos] for k, v in metrics_aggregated.items()}

        self.history.add_loss_distributed(server_round=server_round, loss=loss_aggregated)
        self.history.add_metrics_distributed(server_round=server_round, metrics=metrics_aggregated)
        for proxy, res in results:
            self._last_distr[self._cid_to_index[proxy.cid]] = res.loss

        return loss_aggregated, metrics_aggregated
