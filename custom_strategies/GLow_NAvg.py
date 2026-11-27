"""
Neighbour averaging (strategy: 'navg'), design by Andrea De Luna. Evaluation
and outputs as in GLow_gossip.

Each round r:
 1. Apply the runtime schedule for round index r-1.
 2. Head = (r-1) % num_clients. A 'down' head wastes the round; an 'sl' head
    trains alone.
 3. Participants: the head and its up neighbours. train_mode 'all': all of
    them train; 'head_only': only the head, the rest send stored parameters.
 4. Outside warm-up, the train-sample-weighted average goes to the head and,
    with `share_with_neighbors`, to every participant listing the head in its
    topology row (an isolated malicious/random node never receives it).
"""

import time
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from custom_strategies.GLow_gossip import GLow_gossip


class GLow_NAvg(GLow_gossip):
    """Rotating-head neighbour averaging. See module docstring."""

    name = 'navg'

    def __init__(self, *, share_with_neighbors: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.share_with_neighbors = share_with_neighbors

    def __repr__(self) -> str:
        return f"GLow_NAvg(train_mode={self.train_mode}, share_with_neighbors={self.share_with_neighbors})"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Apply runtime schedule, rotate head, pick participants and trainers."""
        self.current_round = server_round - 1
        self.head_check()
        self._round_post = {}
        self._round_participants = []
        self._round_exchanges = 0

        head = (server_round - 1) % self.min_available_clients
        self.selected_head = head
        status = self.head_status[head]
        if status == 'down':  # wasted round
            return []
        if status == 'sl':
            participants = [head]
        else:
            participants = sorted({head} | {n for n in self.topology[head] if self.head_status[n] == 'up'})
        self._round_participants = participants

        trainers = participants if self.train_mode == 'all' else [head]
        return self._fit_instructions(server_round, trainers, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Store trained parameters, evaluate (pre), average, evaluate (post)."""
        if not self.accept_failures and failures:
            return None, {}

        results = self._store_fit_results(results)
        head, participants = self.selected_head, self._round_participants
        pre = {n: self._eval_node(server_round, n) for n in participants}

        in_warmup = self.warmup_rounds is not None and server_round <= self.warmup_rounds
        if not in_warmup and len(participants) > 1:
            start = time.perf_counter()
            weights = {n: parameters_to_ndarrays(self.head_parameters[n]) for n in participants}
            averaged = ndarrays_to_parameters(self._weighted_average(participants, weights))
            receivers = [n for n in participants if n == head or head in self.topology[n]] if self.share_with_neighbors else [head]
            for n in receivers:
                self.head_parameters[n] = averaged
            self.aggregation_time[head] = (time.perf_counter() - start) * 1000
            self._round_exchanges = 1

        post = {n: self._eval_node(server_round, n) for n in self._active()}
        self._round_post = post
        self._write_tracking(server_round, participants, pre, post)

        # No global model in decentralized learning
        return None, self._record_fit_metrics(server_round, results)
