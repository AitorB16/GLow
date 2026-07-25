"""The Flower client side of GLow's gossip learning: one FlowerClient per
topology node. Every node runs the same client code; GLow_strategy tells each
one, per round, whether it's this round's "head" or one of its "neighbours"
(see FlowerClient.fit()) -- there's no separate client/server role split like
centralized FL has.
"""

from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import Context, NDArrays, Scalar

import os
import json
import torch
import numpy as np
import flwr as fl
from model import LeNet, train, test, compute_prob_matrix

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") #Make GPU run deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, seed, device):
        node_seed = seed #every node starts from the same seed (same initial weights)
        np.random.seed(node_seed)
        torch.manual_seed(node_seed)

        super().__init__()
        self.cid = int(cid)
        self.validation_loaders = validationloader
        self.trainloader = trainloader[cid]
        self.validationloader = validationloader[cid]
        self.local_acc = None
        self.model = LeNet(num_classes)
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() and (device == 'GPU' or device == 'H100') else "cpu")
        #self.device = torch.device("mps")

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

    def fit(self, parameters, config):
        """Called once per round for every up-neighbour of this round's head
        (`config['head_cid']`), including the head itself.
        """
        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)

        if self.trainloader != '':
            self.trainloader.generator.manual_seed(config['seed'])

        class_client_matrix = {self.cid: self.val_counts}

        # Determine number of epochs (malicious/else branches are identical --
        # no actual training-length difference for malicious clients here)
        if int(config['comm_round']) <= config['warmup_rounds']:
            epochs = config['warmup_epochs']
            print(f" -> Client {self.cid}: Warm-up Phase Active! Training for {epochs} epochs.")
        elif config['nature'] == 'malicious':
            epochs = config['local_epochs']
        else:
            epochs = config['local_epochs']

        # Only the head trains this round -- neighbours just re-evaluate their
        # existing stored parameters (below) and report those, untouched.
        lr = config['lr']
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        metrics_val_distr = 0.
        centroid = [0] * self.num_classes

        #print("======TRAINING==== cid:", self.cid, "====head_cid:", config['head_cid'])
        if config['head_cid'] == self.cid:
            _, metrics_val_distr, centroid = train(
                self.model, self.trainloader, self.validationloader,
                optim, epochs, self.num_classes, config['nature'], self.device
            )

        # Prob matrix always evaluated on head validation set
        prob_matrix, _ = compute_prob_matrix(
            self.model, self.validation_loaders[config['head_cid']],
            self.num_classes, config['nature'], self.device
        )

        # Head client branch
        if config['head_cid'] == self.cid:
            #print("======HERE==== HEAD", config['head_cid'], "====cid===", self.cid)
            return self.get_parameters({}), len(self.trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'centroid': json.dumps(centroid.tolist()),
                'confidence_score': json.dumps(centroid.tolist()),
                'prob_matrix': json.dumps(prob_matrix.flatten().tolist()),
                'HEAD': 'YES',
                'distr_val_loss': '##',
                'energy used': '10W'
            }

        # Neighbour clients branch
        elif self.cid in json.loads(config['neighbors']):
            # Step 1: neighbour's own centroid
            _, _, neighbour_centroid, _ = test(
                self.model, self.validation_loaders[self.cid],
                self.num_classes, config['nature'], self.device
            )

            # Step 2: head's centroid
            _, _, head_centroid, _ = test(
                self.model, self.validation_loaders[config['head_cid']],
                self.num_classes, config['nature'], self.device
            )

            # Unwrap if they are lists of tensors
            if isinstance(neighbour_centroid, list) and len(neighbour_centroid) > 0:
                neighbour_centroid = neighbour_centroid[0]
            if isinstance(head_centroid, list) and len(head_centroid) > 0:
                head_centroid = head_centroid[0]

            # Convert to numpy arrays
            neighbour_centroid = neighbour_centroid.detach().cpu().numpy()
            head_centroid = head_centroid.detach().cpu().numpy()

            # Step 3: rebuild confidence score
            confidence_score = []
            for class_id, c in enumerate(neighbour_centroid):
                neighbour_has_class = class_client_matrix[self.cid][class_id] > 0
                if not neighbour_has_class:
                    confidence_score.append(float(head_centroid[class_id]))
                else:
                    confidence_score.append(float(c))

            confidence_score = np.array(confidence_score)
            # Compare neighbour_centroid vs confidence_score element-wise
            diffs = [(i, float(neighbour_centroid[i]), float(confidence_score[i]))
                    for i in range(len(neighbour_centroid))
                    if neighbour_centroid[i] != confidence_score[i]]

            # Return neighbour metrics
            return self.get_parameters({}), len(self.trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'centroid': json.dumps([]),
                'confidence_score': json.dumps(confidence_score.tolist()),
                'prob_matrix': json.dumps(prob_matrix.flatten().tolist()),
                'HEAD': 'NO',
                'distr_val_loss': '##',
                'energy used': '10W'
            }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Distributed-evaluation phase (separate from fit()'s training
        round): loads `parameters` as-is (no local training) and scores them
        against this client's own validation set -- used by
        GLow_strategy.aggregate_evaluate() to track each node's standalone
        accuracy/F1 over time, independent of who was head."""
        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)
        loss, accuracy, _, macro_f1 = test(self.model, self.validationloader, self.num_classes, config['nature'], self.device)
        return float(loss), len(self.validationloader), {'acc_distr': accuracy, 'macro_f1': macro_f1 , 'cid': self.cid} #send anything, time it took to evaluation, memory usage...

def generate_client_fn(cids, trainloaders, validationloaders, num_classes, seed, device):
    """Factory Flower calls once per simulated node to construct its
    FlowerClient, via the returned `client_fn(context)` closure."""
    def client_fn(context: Context):
        # Current Flower identifies simulated clients via context.node_config
        # (an opaque per-run node id), not a small sequential int -- the
        # "partition-id" entry is what the simulation backend assigns
        # deterministically per client, and is the direct replacement for the
        # old `cid: str` argument this factory used to receive.
        partition_id = int(context.node_config["partition-id"])
        return FlowerClient(cids[partition_id], trainloader=trainloaders, validationloader=validationloaders, num_classes=num_classes, seed=seed, device=device).to_client()
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