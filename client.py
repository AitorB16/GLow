from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Context, NDArrays, Scalar

from model import LeNet, test, train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, seed, device):
        np.random.seed(seed)
        torch.manual_seed(seed)

        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.local_acc = None
        self.model = LeNet(num_classes)
        self.num_classes = num_classes
        self.seed = seed  # Store seed for use in fit/evaluate
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and (device == "GPU" or device == "H100")
            else "cpu"
        )

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_local_acc(self, acc):
        self.local_acc = acc

    def get_local_acc(self):
        return self.local_acc

    def fit(self, parameters, config):
        # Use instance seed instead of config['seed'] for Flower 1.25.0 compatibility
        torch.manual_seed(self.seed)

        # copy params from server in local models
        self.set_parameters(parameters)
        metrics_val_distr = None

        # Perform local training just in the selected node head
        if (
            config["local_train_cid"] == self.cid or config["local_train_cid"] == -1
        ):  # Case for GL or Case for FL
            lr = config["lr"]
            if (
                config["comm_round"] <= config["num_agents"]
            ):  # In first n initial rounds
                epochs = config[
                    "local_epochs"
                ]  # Option to achieve a faster converge in the first * 3 epochs
            elif config["nature"] == "malicious":
                epochs = config["local_epochs"] * 5
            else:
                epochs = config["local_epochs"]

            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            # local training
            distr_loss_train, metrics_val_distr = train(
                self.model,
                self.trainloader,
                self.validationloader,
                optim,
                epochs,
                self.num_classes,
                config["nature"],
                self.device,
            )

            # Convert numpy types to Python types for Flower 1.25.0 compatibility
            return (
                self.get_parameters({}),
                len(self.trainloader),
                {
                    "acc_val_distr": metrics_val_distr,
                    "cid": int(self.cid),
                    "HEAD": "YES",
                    "distr_val_loss": "##",
                    "energy used": "10W",
                },
            )

        # Return current acc and params from neighbours
        # Convert numpy types to Python types for Flower 1.25.0 compatibility
        return (
            self.get_parameters({}),
            len(self.trainloader),
            {
                "acc_val_distr": self.local_acc,
                "cid": int(self.cid),
                "HEAD": "NO",
                "distr_val_loss": "##",
                "energy used": "10W",
            },
        )

    # Evaluate global model in validation set of a particular client
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # Use instance seed instead of config['seed'] for Flower 1.25.0 compatibility
        torch.manual_seed(self.seed)

        self.set_parameters(parameters)
        loss, accuracy = test(
            self.model,
            self.validationloader,
            self.num_classes,
            config["nature"],
            self.device,
        )
        # Convert numpy types to Python types for Flower 1.25.0 compatibility
        return (
            float(loss),
            len(self.validationloader),
            {"acc_distr": accuracy, "cid": int(self.cid)},
        )  # send anything, time it took to evaluation, memory usage...


def generate_client_fn(
    vcid, trainloaders, validationloaders, num_classes, seed, device
):
    def client_fn(context: Context):
        # In Flower 1.25.0, the partition ID is in context.node_config
        # It's a string, so we need to convert it to int
        partition_id = int(context.node_config["partition-id"])

        # Ensure partition_id is within valid range
        if partition_id < 0 or partition_id >= len(trainloaders):
            raise ValueError(
                f"Invalid partition ID: {partition_id}, must be between 0 and {len(trainloaders) - 1}"
            )

        return FlowerClient(
            vcid[partition_id],  # Use vcid mapping for the client's virtual ID
            trainloader=trainloaders[partition_id],
            validationloader=validationloaders[partition_id],
            num_classes=num_classes,
            seed=seed,
            device=device,
        ).to_client()

    return client_fn


def cli_eval_distr_results(
    metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m["acc_distr"])
        vcid.append(m["cid"])
    # Aggregate and return custom metric (weighted average)
    return {"acc_distr": acc, "cid": vcid}


def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m["acc_val_distr"])
        vcid.append(m["cid"])

    # Aggregate and return custom metric (weighted average)
    return {"acc_val_distr": acc, "cid": vcid}
