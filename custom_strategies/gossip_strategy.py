import flwr as fl

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

from .aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from .strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

class GLStrategy(Strategy):
    def initialize_parameters(self, client_manager):
        # Your implementation here

    def configure_fit(self, server_round, parameters, client_manager):
        # Your implementation here

    def aggregate_fit(self, server_round, results, failures):
        # Your implementation here

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Your implementation here

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here

    def evaluate(self, parameters):
        # Your implementation here


#class CustomClientConfigStrategy(fl.server.strategy.FedAvg):
#    def configure_fit(
#        self, server_round: int, parameters: Parameters, client_manager: ClientManager
#    ) -> List[Tuple[ClientProxy, FitIns]]:
#        client_instructions = super().configure_fit(server_round, parameters, client_manager)
#
#        # Add special "hello": "world" config key/value pair,
#        # but only to the first client in the list
#        _, fit_ins = client_instructions[0]  # First (ClientProxy, FitIns) pair
#        fit_ins.config["hello"] = "world"  # Change config for this client only
#
#        return client_instructions