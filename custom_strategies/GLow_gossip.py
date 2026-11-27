"""
Pairwise gossip (strategy: 'gossip'), design by Andrea De Luna.

Each round r:
 1. Apply the runtime schedule for round index r-1 (as in vanilla).
 2. train_mode 'all': every non-down node trains; 'head_only': one random
    non-down node trains.
 3. Outside warm-up, `num_gossip_iterations` attempts: draw an up node A and an
    up neighbour B from topology[A]; A takes the sample-weighted average of A
    and B, and so does B if A is in topology[B] (an isolated malicious/random
    node can be pulled from but never pulls).
 4. Evaluate every non-down node before and after gossip, on its own and on the
    global test set (<run_id>_gossip_tracking.csv).

'sl' nodes train but never gossip; 'down' nodes do nothing.
"""

import csv
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from custom_strategies.GLow_vanilla import GLow_vanilla


class _CidCriterion(Criterion):
    def __init__(self, cid_list):
        self.cid_list = cid_list

    def select(self, client: ClientProxy) -> bool:
        return client.cid in self.cid_list


class GLow_gossip(GLow_vanilla):
    """See module docstring."""

    name = 'gossip'

    def __init__(
        self,
        *,
        train_mode: str = 'all',
        num_gossip_iterations: int = 500,
        evaluate_global_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if train_mode not in ('all', 'head_only'):
            raise ValueError(f"Unknown train_mode '{train_mode}'. Expected 'all' or 'head_only'.")
        self.train_mode = train_mode
        self.num_gossip_iterations = num_gossip_iterations
        self.evaluate_global_fn = evaluate_global_fn
        # own RNG for head and pair draws
        self.rng = np.random.default_rng(self.seed)
        # averaging weights: train sample count per node
        self.num_samples = [int(np.sum(row)) for row in self.class_client_matrix_train]

        self._round_post = {}       # node -> post-gossip eval of the current round
        self._eval_cache = {}       # node -> (Parameters object, nature, eval result)
        self._last_central = None   # last (loss, metrics) added to history
        self._round_participants = []
        self._round_exchanges = 0

        self.csv_path = f'{self.save_path}{self.run_id}_gossip_tracking.csv'
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(["seed", "round", "client_cid", "stage", "loss_local_test", "acc_local_test", "f1_local_test", "loss_global_test", "acc_global_test", "f1_global_test"])

    def __repr__(self) -> str:
        return f"GLow_gossip(train_mode={self.train_mode}, num_gossip_iterations={self.num_gossip_iterations})"

    def _active(self) -> List[int]:
        """Nodes that take part in the round at all (train/evaluate)."""
        return [n for n in range(self.min_available_clients) if self.head_status[n] != 'down']

    def _gossipers(self) -> List[int]:
        """Nodes allowed to exchange parameters ('sl' and 'down' excluded)."""
        return [n for n in range(self.min_available_clients) if self.head_status[n] == 'up']

    def _eval_node(self, server_round: int, node: int) -> Dict:
        """Test `node` on its own and the global test set; cached while its
        parameters and nature are unchanged."""
        params, nature = self.head_parameters[node], self.head_nature[node]
        cached = self._eval_cache.get(node)
        if cached is not None and cached[0] is params and cached[1] == nature:
            return cached[2]

        ndarrays = parameters_to_ndarrays(params)
        config = {'nature': nature, 'seed': self._round_seed(server_round, node)}
        loss, metrics = self.evaluate_fn(node, server_round, ndarrays, config)
        res = {'loss': loss, 'acc': metrics['acc_test'], 'f1': metrics['macro_f1_test'],
               'preds_per_class': metrics['preds_per_class'], 'loss_g': 0.0, 'acc_g': 0.0, 'f1_g': 0.0}
        if self.evaluate_global_fn is not None:
            loss_g, metrics_g = self.evaluate_global_fn(node, server_round, ndarrays, config)
            res['loss_g'], res['acc_g'], res['f1_g'] = loss_g, metrics_g['acc_test'], metrics_g['macro_f1_test']

        self._eval_cache[node] = (params, nature, res)
        return res

    def _weighted_average(self, nodes: List[int], weights: Dict[int, List]):
        """Train-sample-weighted average (plain mean if no node has data)."""
        counts = np.array([self.num_samples[n] for n in nodes], dtype=float)
        scale = counts / counts.sum() if counts.sum() > 0 else np.full(len(nodes), 1. / len(nodes))
        return [sum(s * layers[i] for s, layers in zip(scale, (weights[n] for n in nodes)))
                for i in range(len(weights[nodes[0]]))]

    def _gossip(self, gossipers: List[int]) -> int:
        """Random pairwise exchanges; returns how many happened."""
        if len(gossipers) < 2:
            return 0
        weights = {n: parameters_to_ndarrays(self.head_parameters[n]) for n in gossipers}
        gossip_set = set(gossipers)
        touched = set()
        exchanges = 0
        for _ in range(self.num_gossip_iterations):
            a = int(self.rng.choice(gossipers))
            neighbours = [n for n in self.topology[a] if n in gossip_set and n != a]
            if not neighbours:
                continue
            b = int(self.rng.choice(neighbours))
            avg = self._weighted_average([a, b], weights)
            weights[a] = avg
            touched.add(a)
            if a in self.topology[b]:
                weights[b] = avg
                touched.add(b)
            exchanges += 1
        for n in touched:
            self.head_parameters[n] = ndarrays_to_parameters(weights[n])
        return exchanges

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Apply the schedule and pick the trainers."""
        self.current_round = server_round - 1
        self.head_check()
        self._round_post = {}
        self._round_participants = []
        self._round_exchanges = 0

        active = self._active()
        if not active:
            self.selected_head = -1
            return []

        # -1 = no single head (every active node trains)
        self.selected_head = int(self.rng.choice(active)) if self.train_mode == 'head_only' else -1
        trainers = [self.selected_head] if self.train_mode == 'head_only' else active
        return self._fit_instructions(server_round, trainers, client_manager)

    def _fit_instructions(self, server_round: int, trainers: List[int], client_manager: ClientManager):
        """One FitIns per trainer with its own stored parameters."""
        clients = client_manager.sample(
            num_clients=len(trainers), criterion=_CidCriterion(self._cids_for_indices(trainers))
        )
        pairs = []
        for client in clients:
            index = self._cid_to_index[client.cid]
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            config['strategy'] = self.name
            config['head_cid'] = self.selected_head
            config['comm_round'] = server_round
            config['warmup_rounds'] = self.warmup_rounds
            config['warmup_epochs'] = self.warmup_epochs
            config['nature'] = self.head_nature[index]  # each node trains with its own nature
            config['seed'] = self._round_seed(server_round, index)
            pairs.append((client, FitIns(self.head_parameters[index], config)))
        return pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Store trained parameters, evaluate (pre), gossip, evaluate (post)."""
        if not self.accept_failures and failures:
            return None, {}

        results = self._store_fit_results(results)

        active = self._active()
        pre = {n: self._eval_node(server_round, n) for n in active}

        in_warmup = self.warmup_rounds is not None and server_round <= self.warmup_rounds
        gossipers = [] if in_warmup else self._gossipers()
        start = time.perf_counter()
        self._round_exchanges = self._gossip(gossipers)
        aggregation_time = (time.perf_counter() - start) * 1000
        self._round_participants = gossipers
        for n in gossipers:
            self.aggregation_time[n] = aggregation_time

        post = {n: self._eval_node(server_round, n) for n in active}
        self._round_post = post
        self._write_tracking(server_round, active, pre, post)

        # No global model in gossip learning
        return None, self._record_fit_metrics(server_round, results)

    def _store_fit_results(self, results):
        # sorted: completion order varies with Ray
        results = sorted(results, key=lambda r: self._cid_to_index[r[0].cid])
        for proxy, fit_res in results:
            self.head_parameters[self._cid_to_index[proxy.cid]] = fit_res.parameters
        return results

    def _write_tracking(self, server_round: int, nodes: List[int], pre: Dict, post: Dict):
        """PRE/POST aggregation rows for `nodes` in <run_id>_gossip_tracking.csv."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for n in nodes:
                for stage, r in (('PRE_GOSSIP', pre[n]), ('POST_GOSSIP', post[n])):
                    writer.writerow([self.seed, server_round, n, stage, float(r['loss']), float(r['acc']), float(r['f1']), float(r['loss_g']), float(r['acc_g']), float(r['f1_g'])])

    def _record_fit_metrics(self, server_round: int, results) -> Dict:
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn and results:
            metrics_aggregated = self.fit_metrics_aggregation_fn([(res.num_examples, res.metrics) for _, res in results])
            for cid, t in zip(metrics_aggregated['cid'], metrics_aggregated['training_time']):
                self.training_time[cid] = t
            self.history.add_metrics_distributed_fit(server_round=server_round, metrics=metrics_aggregated)
        return metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Every active node validates its post-gossip parameters."""
        if self.fraction_evaluate == 0.0:
            return []
        active = self._active()
        if not active:
            return []
        clients = client_manager.sample(
            num_clients=len(active), criterion=_CidCriterion(self._cids_for_indices(active))
        )
        pairs = []
        for client in clients:
            index = self._cid_to_index[client.cid]
            config = {}
            if self.on_evaluate_config_fn is not None:
                config = self.on_evaluate_config_fn(server_round)
            config['head_cid'] = self.selected_head
            config['nature'] = self.head_nature[index]
            config['seed'] = self._round_seed(server_round, index)
            pairs.append((client, EvaluateIns(self.head_parameters[index], config)))
        return pairs

    def _val_loss(self, results) -> float:
        """losses_val: mean over nodes with a validation set, as losses_test."""
        return float(np.mean([res.loss for _, res in results if res.num_examples > 0]))

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """History test entry: mean over active nodes of their post-gossip test metrics."""
        if self.evaluate_fn is None:
            return None

        if server_round == 0:  # initial models
            self._round_post = {n: self._eval_node(0, n) for n in self._active()}
        else:
            self.round_log.append((server_round, self.selected_head, list(self._round_participants)))

        post = self._round_post
        if not post:  # every node down: repeat the last values
            if self._last_central is not None:
                loss, metrics = self._last_central
                self.history.add_loss_centralized(server_round=server_round, loss=loss)
                self.history.add_metrics_centralized(server_round=server_round, metrics=metrics)
            self._log_round(server_round, post)
            return None

        for n, r in post.items():
            self.head_losses[n] = r['loss']
            self.head_metrics[n] = r['acc']
            self.head_f1[n] = r['f1']
            self.head_preds_per_class[n] = r['preds_per_class']

        loss = float(np.nanmean([r['loss'] for r in post.values()]))
        metrics = {'acc_test': float(np.nanmean([r['acc'] for r in post.values()])),
                   'macro_f1_test': float(np.nanmean([r['f1'] for r in post.values()]))}
        self.history.add_loss_centralized(server_round=server_round, loss=loss)
        self.history.add_metrics_centralized(server_round=server_round, metrics=metrics)
        self._last_central = (loss, metrics)

        self._log_round(server_round, post)
        return loss, metrics

    def _log_round(self, server_round: int, post: Dict[int, Dict]):
        if server_round == 0:
            return
        num = lambda v: '-' if v is None else f'{v:.3f}'
        for n in range(self.min_available_clients):
            r = post.get(n)
            self.out_buf += (f'node_ID: {n} round ID: {server_round} head: {self.selected_head} '
                             f'exchanges: {self._round_exchanges} '
                             f'loss_local_test: {num(r and r["loss"])} acc_local_test: {num(r and r["acc"])} f1_local_test: {num(r and r["f1"])} '
                             f'loss_global_test: {num(r and r["loss_g"])} acc_global_test: {num(r and r["acc_g"])} f1_global_test: {num(r and r["f1_g"])} '
                             f'training_time: {self.training_time[n]} '
                             f'status: {self.head_status[n]}\n')
        mean = lambda k: num(np.nanmean([r[k] for r in post.values()])) if post else '-'
        print(f'round {server_round}/{self.total_rounds}  head {self.selected_head}  '
              f'participants {len(self._round_participants)}  exchanges {self._round_exchanges}  '
              f'mean loss_local_test {mean("loss")}  mean acc_local_test {mean("acc")}  mean f1_local_test {mean("f1")}  '
              f'mean loss_global_test {mean("loss_g")}  mean acc_global_test {mean("acc_g")}  mean f1_global_test {mean("f1_g")}', flush=True)
        if server_round == self.total_rounds:
            self.save_on_finish()
