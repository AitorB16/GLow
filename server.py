from collections import OrderedDict

import torch

from model import LeNet, test


def get_on_fit_config(config):
    def fit_config_fn(server_round: int):
        """Decrease the learning rate from a specific communication round on"""
        # if server_round > 50:
        #    lr = config['lr'] / 10
        # else:
        #    lr = config['lr']
        lr = config["lr"]
        return {
            "lr": lr,
            #'momentum': config['momentum'],
            "local_epochs": config["local_epochs"],
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloaders, seed: int = 1999):

    def evaluate_fn(sid: int, server_round: int, parameters, config):  # int nparrays, dict
        # Use passed seed parameter instead of config['seed'] for Flower 1.25.0 compatibility
        torch.manual_seed(seed)

        model = LeNet(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloaders[sid], num_classes, config["nature"], device)  # global model
        return loss, {"acc_cntrl": accuracy}

    return evaluate_fn
