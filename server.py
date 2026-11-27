"""Fit-config and test-evaluation callbacks passed to the strategy by main.py."""

from collections import OrderedDict
from models import build_model, test, compute_prob_matrix
import torch

def get_on_fit_config(config):
    optimizer = config['optimizer']
    if optimizer not in ('adam', 'sgd'):
        raise ValueError(f"Unknown optimizer '{optimizer}'. Expected 'adam' or 'sgd'.")

    def fit_config_fn(server_round: int):
        # lr schedule, e.g.: lr = config['lr'] / 10 if server_round > 50 else config['lr']
        lr = config['lr']
        return {'lr': lr,
                'optimizer': optimizer,
                'momentum': config['momentum'],
                'local_epochs': config['local_epochs']
                }

    return fit_config_fn

def get_evaluate_fn(num_classes: int, testloaders, dataset: str):
    """evaluate_fn(sid, ...) scores `parameters` on node `sid`'s test set."""

    def evaluate_fn(sid: int, server_round: int, parameters, config):
        torch.manual_seed(config['seed'])

        model = build_model(dataset, num_classes)
        device = torch.device("cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, _, macro_f1 = test(model, testloaders[sid], num_classes, config['nature'], device)
        _, preds_per_class = compute_prob_matrix(model, testloaders[sid], num_classes, config['nature'], device)

        return loss, {'acc_test': accuracy, 'macro_f1_test': macro_f1, 'preds_per_class': preds_per_class.cpu().numpy()}

    return evaluate_fn
