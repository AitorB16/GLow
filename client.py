"""The Flower client side of GLow's gossip learning: one FlowerClient per
topology node. Every node runs the same client code; GLow_vanilla tells each
one, per round, whether it's this round's "head" or one of its "neighbours"
(see FlowerClient.fit()) -- there's no separate client/server role split like
centralized FL has.
"""

import copy
from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import Context, NDArrays, Scalar

import json
import torch
import numpy as np
import flwr as fl
from models import build_model, train, test, compute_prob_matrix, softmax_distance, unpack_ndarrays, load_weights

torch.use_deterministic_algorithms(True, warn_only=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, dataset, seed):
        node_seed = seed #every node starts from the same seed (same initial weights)
        np.random.seed(node_seed)
        torch.manual_seed(node_seed)

        super().__init__()
        self.cid = int(cid)
        self.validation_loaders = validationloader
        self.trainloader = trainloader[cid]
        self.validationloader = validationloader[cid]
        self.local_acc = None
        self.model = build_model(dataset, num_classes)
        self.num_classes = num_classes
        self.device = torch.device("cpu")

        self.val_counts = [0] * self.num_classes
        for _, labels in self.validationloader:
            for c in range(self.num_classes):
                self.val_counts[c] += (labels == c).sum().item()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_local_acc(self, acc):
        self.local_acc = acc

    def get_local_acc(self):
        return self.local_acc

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return {"partition_id": self.cid}

    def _make_optimizer(self, config):
        """Local optimizer from config_fit's `optimizer` ('adam', or 'sgd'
        with config_fit's `momentum`)."""
        if config['optimizer'] == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
        return torch.optim.Adam(self.model.parameters(), lr=config['lr'])

    @staticmethod
    def _num_samples(loader):
        """Sample count, not batch count -- len(DataLoader) is the number of
        batches, and rounding up over-weights small shards. Dataless clients
        hold '' instead of a loader (see dataset.py)."""
        return len(loader.dataset) if loader != '' else 0

    def fit(self, parameters, config):
        """Called once per round for every up-neighbour of this round's head
        (`config['head_cid']`), including the head itself.
        """
        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)

        if self.trainloader != '':
            self.trainloader.generator.manual_seed(config['seed'])

        if int(config['comm_round']) <= config['warmup_rounds']:
            epochs = config['warmup_epochs']
            #print(f" -> Client {self.cid}: Warm-up Phase Active! Training for {epochs} epochs.")
        else:
            epochs = config['local_epochs']

        if config['strategy'] in ('gossip', 'navg'):
            return self._fit_gossip(config, epochs)

        # Only the head trains this round -- neighbours keep their stored
        # parameters untouched and only measure them (epochs=0 runs train()'s
        # validation pass alone, the same code path the head's accuracy and
        # recall come from).
        is_head = config['head_cid'] == self.cid
        aggregation = config['aggregation']
        nature = config['nature']

        optim = self._make_optimizer(config) if is_head else None
        _, metrics_val_distr, recall_own, training_time = train(
            self.model, self.trainloader, self.validationloader,
            optim, epochs if is_head else 0, self.num_classes, nature, self.device
        )
        if not is_head:
            training_time = 0.
        if self._num_samples(self.validationloader) == 0:
            metrics_val_distr = 0.

        metrics = {
            'acc_val_distr': metrics_val_distr,
            'cid': self.cid,
            'centroid': json.dumps(recall_own.tolist() if is_head else []),
            'prob_matrix': json.dumps([]),
            'training_time': training_time,
            'HEAD': 'YES' if is_head else 'NO',
        }

        # Raw per-class signals for the approach_2 / approach_2b aggregators,
        # which combine them (see flwr_lib_modifications/aggregate.py).
        if aggregation in ('approach_2', 'approach_2b'):
            head_val = self.validation_loaders[config['head_cid']]
            recall_head = recall_own if is_head else test(self.model, head_val, self.num_classes, nature, self.device)[2]
            metrics['recall_own'] = json.dumps(recall_own.tolist())
            metrics['recall_head'] = json.dumps(recall_head.tolist())
            metrics['own_val_has'] = json.dumps([count > 0 for count in self.val_counts])
        if aggregation == 'approach_2':
            prob_matrix, _ = compute_prob_matrix(self.model, head_val, self.num_classes, nature, self.device)
            metrics['prob_matrix'] = json.dumps(prob_matrix.flatten().tolist())
        if aggregation == 'approach_2b' and not is_head:
            head_model = copy.deepcopy(self.model)
            load_weights(head_model, unpack_ndarrays(config['head_params']))
            metrics['dist_N'] = json.dumps(softmax_distance(self.model, head_model, self.validationloader, self.num_classes, nature, self.device))

        return self.get_parameters({}), self._num_samples(self.trainloader), metrics

    def _fit_gossip(self, config, epochs):
        """GLow_gossip / GLow_NAvg fit() reply: plain local training on this node's own
        data (its own `nature`); gossip itself happens in the strategy.
        Returns the keys cli_val_distr() expects."""
        optim = self._make_optimizer(config)
        _, metrics_val_distr, _, training_time = train(
            self.model, self.trainloader, self.validationloader,
            optim, epochs, self.num_classes, config['nature'], self.device
        )
        if self._num_samples(self.validationloader) == 0:
            metrics_val_distr = 0.
        return self.get_parameters({}), self._num_samples(self.trainloader), {
            'acc_val_distr': metrics_val_distr,
            'cid': self.cid,
            'centroid': json.dumps([]),
            'prob_matrix': json.dumps([]),
            'training_time': training_time,
            'HEAD': 'YES' if config['head_cid'] == self.cid else 'NO',
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Distributed-evaluation phase (separate from fit()'s training
        round): loads `parameters` as-is (no local training) and scores them
        against this client's own validation set -- used by
        GLow_vanilla.aggregate_evaluate() to track each node's standalone
        accuracy/F1 over time, independent of who was head."""
        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)
        loss, accuracy, _, macro_f1 = test(self.model, self.validationloader, self.num_classes, config['nature'], self.device)
        return float(loss), self._num_samples(self.validationloader), {'acc_distr': accuracy, 'macro_f1': macro_f1 , 'cid': self.cid} #send anything, time it took to evaluation, memory usage...

def generate_client_fn(cids, trainloaders, validationloaders, num_classes, dataset, seed):
    """Factory Flower calls once per simulated node to construct its
    FlowerClient, via the returned `client_fn(context)` closure."""
    def client_fn(context: Context):
        # Current Flower identifies simulated clients via context.node_config NEED MAPPING
        partition_id = int(context.node_config["partition-id"])
        return FlowerClient(cids[partition_id], trainloader=trainloaders, validationloader=validationloaders, num_classes=num_classes, dataset=dataset, seed=seed).to_client()
    return client_fn

def cli_eval_distr_results(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    """`evaluate_metrics_aggregation_fn`: collects every client's per-round
    evaluate() metrics into parallel lists rather than averaging them, so
    each client's trajectory can be tracked individually."""
    acc = []
    cids = []
    for num_examples, m in metrics:
        acc.append(m['acc_distr'])
        cids.append(m['cid'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_distr": acc, "cid": cids}

def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    """Same as `cli_eval_distr_results` but for fit()'s validation metrics
    (`acc_val_distr`/`centroid`/`prob_matrix`) -- GLow_vanilla needs each
    client's own values individually for trust-weighted aggregation, not a
    blended average."""
    acc = []
    cids = []
    centroid = []
    prob_matrix = []
    training_time = []
    for num_examples, m in metrics:
        acc.append(m['acc_val_distr'])
        cids.append(m['cid'])
        centroid.append(m['centroid'])
        prob_matrix.append(m['prob_matrix'])
        training_time.append(m['training_time'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_val_distr": acc, "cid": cids, "centroid": centroid, "prob_matrix": prob_matrix, "training_time": training_time}