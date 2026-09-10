# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Rotating-head design: each round one node in `topology` is head
(`select_head`), its up-neighbours train from the head's parameters
(`get_up_neighbors`), `aggregate_fit` folds results back into
`self.head_parameters[head]`. `head_switch_*`/`head_check` schedule nodes
going up/down/malicious per round.
"""

import os
import json
import flwr
import numpy as np
from collections import OrderedDict
import torch
from models import build_model

from logging import WARNING, INFO
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

from flwr_lib_modifications.aggregate import aggregate_inplace, aggregate_score, aggregate_score_validation, aggregate_score_centroids_2, weighted_loss_avg
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
class GLow_strategy(Strategy):
    """Decentralized gossip-learning strategy. https://arxiv.org/abs/2501.10463
    See module docstring for the rotating-head mechanic.
    """

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
        head_status: List[str],
        head_nature: List[str],
        seed: int,
        save_path: str,
        warmup_rounds: Optional[int] = None,
        warmup_epochs: int = 15,
        dataset: str = 'cifar',
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.total_rounds = total_rounds
        self.current_round = 0
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
        self.num_classes = num_classes
        self.class_client_matrix_train = class_client_matrix_train
        self.seed = seed
        self.save_path = save_path
        self.warmup_rounds = warmup_rounds
        self.warmup_epochs = warmup_epochs
        self.dataset = dataset
        self.head_switch_down = head_switch_down
        self.head_switch_up = head_switch_up
        self.head_switch_malicious = head_switch_malicious
        self.head_status = head_status
        self.head_nature = head_nature
        # ClientProxy.cid -> topology index; simulation ID maps to client ID
        self._cid_to_index: Dict[str, int] = {}
        self.history = History()
        # Caches aggregate_fit()'s per-neighbour eval for neighbours
        self._last_eval_results = {}
        # per-node list of neighbour accuracies/params
        self.neigh_metrics = []
        for i in range(min_available_clients):
            self.neigh_metrics.append([])
            for j in range(min_available_clients):
                self.neigh_metrics[i].append(None)

    def _round_seed(self, server_round: int, node_index: int) -> int:
        """Per-(round, node) seed offset -- keeps RNG state independent of
        client scheduling order under worker-process reuse."""
        return self.seed + server_round * 10_000 + node_index

    def _cids_for_indices(self, indices) -> List[str]:
        """cid for each topology index (reverse lookup via _cid_to_index)."""
        index_set = set(indices)
        return [cid for cid, idx in self._cid_to_index.items() if idx in index_set]

    def get_up_neighbors(self):
        """Online neighbours of the current head; includes the head itself."""
        neighbors = self.topology[self.selected_head]
        up_neighbors = []
        for neighbor in neighbors:
            if self.head_status[neighbor] == 'up':
                up_neighbors.append(neighbor)
        return up_neighbors

    def head_check(self):
        """Apply this round's scheduled up/down/malicious transitions."""
        for agent in self.client_list:
            if self.head_switch_up[agent] is not None:
                if self.current_round in self.head_switch_up[agent]:
                    self.head_status[agent] = 'up'
            if self.head_switch_down[agent] is not None:
                if self.current_round in self.head_switch_down[agent]:
                    self.head_status[agent] = 'down'
            if self.head_switch_malicious[agent] is not None:
                if self.current_round in self.head_switch_malicious[agent]:
                    self.head_nature[agent] = 'malicious'
                    self.topology[agent] = [agent]
                    self.head_metrics[agent] = None
                    self.head_losses[agent] = None
                    self.head_f1[agent] = None
                    self.head_preds_per_class[agent] = np.zeros((self.num_classes, self.num_classes),dtype=int)
                    self.neigh_metrics[agent] = [None] * self.min_available_clients
                    self.head_parameters[agent] = self.initial_parameters[agent]

    def select_head(self):
        """Rotate to this round's head (first online node in client_list)
        and advance current_round."""
        search = True
        while search:
            self.selected_head = self.client_list[0]
            self.head_check()
            if self.head_status[self.selected_head] == 'up':
                search = False
            else:
                self.client_list = np.roll(self.client_list, -1).tolist()
        self.client_list = np.roll(self.client_list, -1).tolist()
        self.current_round += 1

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
        """Learn each client's topology index, seed head_parameters, pick
        round 1's head."""
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

        self.select_head()
        initial_parameters = self.initial_parameters[self.selected_head]
        return initial_parameters

    def save_results(self):
        """Write <run_id>_heads.out, _result_matrix.out, and per-node
        _parameters/<id>.pth."""
        out = ''
        for cli_ID in range(self.min_available_clients):
            out = out + 'head_ID: ' + str(cli_ID) + ' neighbours: ' + str(self.topology[cli_ID]) + ' loss: ' + str(self.head_losses[cli_ID]) + ' acc: ' + str(self.head_metrics[cli_ID]) + ' f1: ' +str(self.head_f1[cli_ID]) + '\n'
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

            self.neigh_metrics[self.selected_head][neighbour] = metrics['acc_cntrl']

            if neighbour == self.selected_head:
                self.head_losses[self.selected_head] = loss
                self.head_metrics[self.selected_head] = metrics['acc_cntrl']
                self.head_f1[self.selected_head] = metrics['macro_f1']
                self.head_preds_per_class[self.selected_head] = metrics['preds_per_class']
                head_loss = loss
                head_metrics = metrics

        if head_loss is None:
            raise RuntimeError(
                f"Node {self.selected_head} is head this round but its own topology "
                f"row (topology[{self.selected_head}]) doesn't include its own index "
                "among its up neighbours -- every row must list itself."
            )

        if server_round == self.total_rounds:
            self.save_results()

        self.history.add_loss_centralized(server_round=server_round, loss=head_loss)
        self.history.add_metrics_centralized(server_round=server_round, metrics=head_metrics)

        return head_loss, head_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Rotate head, build one FitIns per up-neighbour from the head's
        parameters."""
        
        self.select_head()
        connections = self.get_up_neighbors()

        class select_criterion(Criterion):
            def __init__(self, cid_list):
                self.cid_list = cid_list
            def select(self, client: ClientProxy) -> bool:
                return client.cid in self.cid_list

        clients = client_manager.sample(
            num_clients=len(connections), criterion=select_criterion(self._cids_for_indices(connections))
        )
        pairs = []
        for client in clients:
            index = self._cid_to_index[client.cid]
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            # neighbour ids JSON-encoded -- legacy config bridge only allows scalars
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

        # Flower returns results in client completion order, which Ray varies
        # run to run -- pin it so aggregation and metrics are reproducible
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
            self.neigh_metrics[self.selected_head][neighbour] = metrics['acc_cntrl']

            if neighbour == self.selected_head:
                self.head_losses[self.selected_head] = loss
                self.head_metrics[self.selected_head] = metrics['acc_cntrl']
                self.head_f1[self.selected_head] = metrics['macro_f1']
                self.head_preds_per_class[self.selected_head] = metrics['preds_per_class']

        # ClientProxy.cid is a simulation id, needs to be mapped to client ID
        results_by_index = {
            self._cid_to_index[cli.cid]: fit_res for cli, fit_res in results
        }

        if self.aggregation == 'inplace':
            aggregated_ndarrays = aggregate_inplace(results_by_index)
        elif self.aggregation == 'score':
            aggregated_ndarrays = aggregate_score(results_by_index, self.neigh_metrics[self.selected_head], up_neighbours, self.selected_head)
        elif self.aggregation == 'score_validation':
            aggregated_ndarrays = aggregate_score_validation(results_by_index, up_neighbours, self.selected_head)
        elif self.aggregation == 'approach_2':
            aggregated_ndarrays = aggregate_score_centroids_2(results_by_index, up_neighbours, self.selected_head, self.current_round, self.class_client_matrix_train, self.num_classes, 0.4, 0.6, 0.)
        else:
            raise ValueError(
                f"Unknown aggregation strategy '{self.aggregation}'. Expected one of: "
                "'inplace', 'score', 'score_validation', 'approach_2'."
            )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Only the head's own parameters are updated
        self.head_parameters[self.selected_head] = parameters_aggregated

        self.history.add_metrics_distributed_fit(server_round=server_round, metrics=metrics_aggregated)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Weighted-average configure_evaluate()'s per-neighbour loss/metrics."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # see aggregate_fit: completion order is not reproducible
        results = sorted(results, key=lambda r: self._cid_to_index[r[0].cid])

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        self.history.add_loss_distributed(server_round=server_round, loss=loss_aggregated)
        self.history.add_metrics_distributed(server_round=server_round, metrics=metrics_aggregated)

        return loss_aggregated, metrics_aggregated
