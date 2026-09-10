"""The Flower client side of GLow's gossip learning: one FlowerClient per
topology node. Every node runs the same client code; GLow_strategy tells each
one, per round, whether it's this round's "head" or one of its "neighbours"
(see FlowerClient.fit()) -- there's no separate client/server role split like
centralized FL has.
"""

from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import Context, NDArrays, Scalar

import json
import torch
import numpy as np
import flwr as fl
from models import build_model, train, test, compute_prob_matrix

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
            print(f" -> Client {self.cid}: Warm-up Phase Active! Training for {epochs} epochs.")
        else:
            epochs = config['local_epochs']

        # Only the head trains this round -- neighbours just re-evaluate their
        # existing stored parameters (below) and report those, untouched.
        is_head = config['head_cid'] == self.cid

        if is_head:
            optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
            _, metrics_val_distr, centroid = train(
                self.model, self.trainloader, self.validationloader,
                optim, epochs, self.num_classes, config['nature'], self.device
            )

        # Prob matrix always evaluated on head validation set
        prob_matrix, _ = compute_prob_matrix(
            self.model, self.validation_loaders[config['head_cid']],
            self.num_classes, config['nature'], self.device
        )

        if is_head:
            centroid_json = json.dumps(centroid.tolist())
            confidence_score_json = centroid_json  # head reports its own centroid as its confidence too
        elif self.cid in json.loads(config['neighbors']):
            confidence_score, metrics_val_distr = self._neighbour_confidence_score(config)
            centroid_json = json.dumps([])
            confidence_score_json = json.dumps(confidence_score.tolist())
        else:
            raise ValueError(
                f"Client {self.cid} was asked to fit() but is neither this round's "
                f"head ({config['head_cid']}) nor listed among its neighbours "
                f"({config['neighbors']}) -- should be unreachable given how "
                "configure_fit() samples clients."
            )

        if self._num_samples(self.validationloader) == 0:
            metrics_val_distr = 0.

        return self.get_parameters({}), self._num_samples(self.trainloader), {
            'acc_val_distr': metrics_val_distr,
            'cid': self.cid,
            'centroid': centroid_json,
            'confidence_score': confidence_score_json,
            'prob_matrix': json.dumps(prob_matrix.flatten().tolist()),
            'HEAD': 'YES' if is_head else 'NO',
            'distr_val_loss': '##',
            'energy used': '10W',
        }

    def _neighbour_confidence_score(self, config):
        """Returns (confidence_score, val_accuracy) for a non-head node.

        Per-class confidence: this neighbour's own centroid where it has local
        validation data for that class, else its performance on the head's
        validation set. Both measured against the SAME (untrained-this-round)
        parameters, since a neighbour doesn't train when it's not head.

        `epochs=0` runs train()'s validation pass and skips the epoch loop, so
        the accuracy a neighbour reports comes from exactly the code path the
        head uses, on its own validation split -- never the test partition,
        which clients are not given at all."""
        _, val_accuracy, neighbour_centroid = train(
            self.model, self.trainloader, self.validationloader,
            None, 0, self.num_classes, config['nature'], self.device
        )
        _, _, head_centroid, _ = test(
            self.model, self.validation_loaders[config['head_cid']],
            self.num_classes, config['nature'], self.device
        )
        neighbour_centroid = neighbour_centroid.detach().cpu().numpy()
        head_centroid = head_centroid.detach().cpu().numpy()
        confidence_score = np.array([
            head_centroid[c] if self.val_counts[c] == 0 else neighbour_centroid[c]
            for c in range(self.num_classes)
        ])
        return confidence_score, val_accuracy

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Distributed-evaluation phase (separate from fit()'s training
        round): loads `parameters` as-is (no local training) and scores them
        against this client's own validation set -- used by
        GLow_strategy.aggregate_evaluate() to track each node's standalone
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
    (`acc_val_distr`/`centroid`/`prob_matrix`) -- GLow_strategy needs each
    client's own values individually for trust-weighted aggregation, not a
    blended average."""
    acc = []
    cids = []
    centroid = []
    prob_matrix = []
    for num_examples, m in metrics:
        acc.append(m['acc_val_distr'])
        cids.append(m['cid'])
        centroid.append(m['centroid'])
        prob_matrix.append(m['prob_matrix'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_val_distr": acc, "cid": cids, "centroid": centroid, "prob_matrix": prob_matrix}