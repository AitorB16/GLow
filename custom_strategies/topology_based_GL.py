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
"""Topology based GL [Belenguer et al., 2025] strategy.

Manuscript: ###############
"""

import os
import flwr
import numpy as np
from collections import OrderedDict
import torch
from model import LeNet

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

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, aggregate_score, aggregate_score_neigh_params, aggregate_median, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from  flwr.server.criterion import Criterion

from flwr.common.typing import GetParametersIns


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class topology_based_Avg(Strategy):
    """Decentralized Averaging strategy.

    Implementation based on https://arxiv.org/abs/2501.10463

    Parameters
    ----------
    total_rounds: int
        Total number of communication rounds for the whole system
    topology: List[List[int]]
        List containing all the node heads with their corresponding list of neighbor nodes
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : List of Parameters, optional
        List of initial model parameters per head.
    pool_parameters: List of Parameters, optional
        List of model parameters per head, updated during the training.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    early_local_train: bool, optional
        SL for a fixed number of local rounds before starting communicating with other neighbors
    inplace: bool. Defaults to True
        Does in-place weighted average of results
    run_id: str
        Run name
    save_path: str
        Path to save achieved results
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        total_rounds: int,
        topology: List[List[int]],
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2, #Varible num subject to topology, (default value) not initialized here
        min_evaluate_clients: int = 2, #Varible num subject to topology, (default value) not initialized here
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
        pool_parameters: Optional[List[Parameters]] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        early_local_train: Optional[bool] = False,
        aggregation: str = 'score_neigh_params',
        run_id: str,
        num_classes: int,
        save_path: str
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.total_rounds = total_rounds
        self.topology = topology
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.client_list = np.arange(min_available_clients)
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.pool_parameters = pool_parameters
        self.selected_pool = None
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.aggregation = 'score_neigh_params'
        self.pool_metrics = [None] * self.min_available_clients
        self.pool_losses = [None] * self.min_available_clients
        self.run_id = run_id
        self.num_classes = num_classes
        self.save_path = save_path
        self.early_local_train = early_local_train
        
        # CREATE STRUCTURE LIST OF LISTS. NEIGHBOURS IN EACH NODE TO STORE LOCAL ACCURACIES AND PARAMS
        self.neigh_metrics = []
        for i in range(min_available_clients):
            self.neigh_metrics.append([])
            for j in topology[i]:
                self.neigh_metrics[i].append(None)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

      
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        '''Custom num clients depending on connection graph'''
        self.min_fit_clients = len(self.topology[self.selected_pool])
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        '''Custom num clients depending on connection graph'''
        self.min_evaluate_clients = len(self.topology[self.selected_pool])
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        clients = client_manager.sample(self.min_available_clients) #Sample all clients
        ins = GetParametersIns(config={})
        
        if self.initial_parameters is None:
            self.initial_parameters = [None] * self.min_available_clients
            self.pool_parameters = [None] * self.min_available_clients

            for client in clients:
                self.initial_parameters[client.cid] = client.get_parameters(ins=ins, timeout=None).parameters
                self.pool_parameters[client.cid] = self.initial_parameters[client.cid]
        
        initial_parameters = self.initial_parameters[0] #Params from first pool for initialization
        self.selected_pool = 0
        return initial_parameters

    def save_results(self):
        out = ''
        for cli_ID in range(self.min_available_clients):
            out = out + 'pool_ID: ' + str(cli_ID) + ' neighbours: ' + str(self.topology[cli_ID]) + ' loss: ' + str(self.pool_losses[cli_ID]) + ' acc: ' + str(self.pool_metrics[cli_ID]) + '\n'
        f = open(self.save_path + str(self.run_id) + "_pool.out", "w")
        f.write(out)
        f.close()
        # save parameters
        param_path = self.save_path + 'parameters/'
        os.makedirs(param_path)
        for cli_ID in range(self.min_available_clients):
            net = LeNet(self.num_classes)
            cli_params_ndarrays = parameters_to_ndarrays(self.pool_parameters[self.selected_pool])
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), cli_params_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(net.state_dict(), param_path + str(cli_ID) + '.pth')

    #EVALUATE INPLACE WEIGHT AVG (SAME TESTSET)
    #def evaluate(
    #    self, server_round: int, parameters: Parameters
    #) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    #    """Evaluate model parameters using an evaluation function."""
    #    if self.evaluate_fn is None:
    #        # No evaluation function provided
    #        return None
    #    parameters_ndarrays = parameters_to_ndarrays(self.pool_parameters[self.selected_pool]) #GET CUSTOM PARAMS
    #    eval_res = self.evaluate_fn(self.selected_pool, server_round, parameters_ndarrays, {}) #CALL CUSTOM FUNC
    #    if eval_res is None:
    #        return None
    #    loss, metrics = eval_res
    #
    #    # Track each pool metrics and results
    #    self.pool_losses[self.selected_pool] = loss
    #    self.pool_metrics[self.selected_pool] = metrics['acc_cntrl']
    #    
    #    # Save pool results and parameters in last rounds
    #    if server_round == self.total_rounds:
    #        self.save_results()
    #
    #    return loss, metrics


    #EVALUATE PARAM PROPAGATION NEIGH (INDEPENDENT TESTSETS)
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        n = 0
        for neighbour in self.topology[self.selected_pool]:
            parameters_ndarrays = parameters_to_ndarrays(self.pool_parameters[neighbour]) #GET CUSTOM PARAMS
            eval_res = self.evaluate_fn(self.selected_pool, server_round, parameters_ndarrays, {}) #CALL CUSTOM FUNC
            
            if eval_res is None:
                return None
            
            loss, metrics = eval_res
            self.neigh_metrics[self.selected_pool][n] = metrics['acc_cntrl']

            # Track each pool metrics and results
            if neighbour == self.selected_pool:
                self.pool_losses[self.selected_pool] = loss
                self.pool_metrics[self.selected_pool] = metrics['acc_cntrl']
                head_loss = loss
                head_metrics = metrics
            n+=1
        
        # Save pool results and parameters in last rounds
        if server_round == self.total_rounds:
            self.save_results()
        
        return head_loss, head_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        self.selected_pool = self.client_list[0] #pick first rotate list
        self.client_list = np.roll(self.client_list, -1).tolist()


        '''Implementing abstract class'''
        class select_criterion(Criterion):
            def __init__(self, cid_list):
                self.cid_list = cid_list
            def select(self, client: ClientProxy) -> bool:
                return client.cid in self.cid_list

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        connections = self.topology[self.selected_pool]

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, criterion=select_criterion(connections)
        )

        # SORT CLIENTS FOR FUTURE AGGREGATIONS
        sorted_clients = []
        for neigh in connections:
            for client in clients:
                if neigh == client.cid:
                    sorted_clients.append(client)

        
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
            config['local_train_cid'] = self.selected_pool
            config['comm_round'] = server_round
            config['num_nodes'] = self.min_available_clients
        pairs = []
        for client in sorted_clients:
            fit_ins = FitIns(self.pool_parameters[client.cid], config)
            pairs.append((client, fit_ins))

        # Return client/config pairs
        return pairs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        
        '''Implementing abstract class'''
        class select_criterion(Criterion):
            def __init__(self, cid_list):
                self.cid_list = cid_list
            def select(self, client: ClientProxy) -> bool:
                return client.cid in self.cid_list     

        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        connections = self.topology[self.selected_pool]

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, criterion=select_criterion(connections)
        )

        # SORT CLIENTS FOR FUTURE AGGREGATIONS
        sorted_clients = []
        for neigh in connections:
            for client in clients:
                if neigh == client.cid:
                    sorted_clients.append(client)

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom fit config function provided
            config = self.on_evaluate_config_fn(server_round)
            config['local_train_cid'] = self.selected_pool
        pairs = []
        for client in sorted_clients:
            evaluate_ins = EvaluateIns(self.pool_parameters[client.cid], config)
            pairs.append((client, evaluate_ins))
        # Return client/config pairs
        return pairs 

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        #Don't aggregate other pool mates in first rounds
        #if self.early_local_train and server_round <= self.min_available_clients:
        #    for client, fit_res in results:
        #        if client.cid != self.selected_pool:
        #            fit_res.num_examples = 0

        if self.aggregation == 'score':
            aggregated_ndarrays = aggregate_score(results, self.pool_metrics, self.topology[self.selected_pool], self.selected_pool)
        elif self.aggregation == 'score_neigh_params':
            aggregated_ndarrays = aggregate_score_neigh_params(results, self.neigh_metrics[self.selected_pool], self.topology[self.selected_pool], self.selected_pool)
        elif self.aggregation == 'inplace':
            # Does in-place weighted average of results
            '''Detect if results are 0'''
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Does weighted average of results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) #fit_res.metrics
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)
            #aggregated_ndarrays = aggregate_median(weights_results)
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.pool_parameters[self.selected_pool] = parameters_aggregated

        '''Spread knowledge to other clients'''
        #No point updating local network parameter of the neighbors with the local average and model

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
