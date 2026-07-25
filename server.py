"""Factories for the two config/eval callbacks GLow_strategy takes as
constructor args (`on_fit_config_fn`, `evaluate_fn`) -- built once in
main.py/hydra_main.py and called internally by the strategy every round.
"""

from collections import OrderedDict
from model import LeNet, test, compute_prob_matrix
import torch

def get_on_fit_config(config):
    """Returns the per-round `FitIns.config` builder GLow_strategy calls in
    configure_fit() (which layers additional keys like `head_cid`/`neighbors`
    on top)."""
    def fit_config_fn(server_round: int):
        '''Decrease the learning rate from a specific communication round on'''
        #if server_round > 50:
        #    lr = config['lr'] / 10
        #else:
        #    lr = config['lr']
        lr = config['lr']
        return {'lr': lr,
                #'momentum': config['momentum'],
                'local_epochs': config['local_epochs']
                }

    return fit_config_fn

def get_evaluate_fn(num_classes: int, testloaders):
    """Returns the `evaluate_fn(sid, server_round, parameters, config)`
    GLow_strategy.evaluate() calls each round to score `parameters` against
    a specific node's own test set (`testloaders[sid]`) -- GLow's
    "centralized"-style evaluation hook, indexed by `sid` since there's no
    single global model in a gossip topology."""

    def evaluate_fn(sid: int, server_round: int, parameters, config): #int nparrays, dict
        torch.manual_seed(config['seed'])

        model = LeNet(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("mps")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, _, macro_f1 = test(model, testloaders[sid], num_classes, config['nature'], device) #global model
        _, preds_per_class = compute_prob_matrix(model, testloaders[sid], num_classes, config['nature'], device) # To compute centroids using neigh params in head val-set

        return loss, {'acc_cntrl': accuracy, 'macro_f1': macro_f1, 'preds_per_class': preds_per_class.cpu().numpy()}

    return evaluate_fn