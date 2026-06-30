from collections import OrderedDict
from model import LeNet, test, compute_prob_matrix
import numpy as np
import torch

def get_on_fit_config(config):
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

        
        #ADD HERE METHOD TO IMPLEMENT WITH VAL/LOADERS? NOPE
        return loss, {'acc_cntrl': accuracy, 'macro_f1': macro_f1, 'preds_per_class': np.array(preds_per_class)}

    return evaluate_fn