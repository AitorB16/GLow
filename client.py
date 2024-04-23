from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar

import torch
import flwr as fl
from model import LeNet, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, ):
        super().__init__()
        
        self.cid = cid

        self.trainloader = trainloader
        self.validationloader = validationloader

        self.model = LeNet()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [ val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        #copy params from server in local models
        self.set_parameters(parameters)

        lr = config['lr']
        #momentum = config['momentum']
        epochs = config['local_epochs']

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        #local training
        distr_loss_train, metrics_val_distr = train(self.model, self.trainloader, self.validationloader, optim, epochs, self.device)
        
        #send how many training instances does a particular client have
        return self.get_parameters({}), len(self.trainloader), {'acc_val_distr': metrics_val_distr,'cid': self.cid, 'energy used': '10W', 'distr_val_loss': '##'}
    
    #Evaluate global model in validation set of a particular client
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.validationloader, self.device)
        return float(loss), len(self.validationloader), {'acc_distr': accuracy, 'cid': self.cid} #send anything, time it took to evaluation, memory usage...
    


def generate_client_fn(vcid, trainloaders, validationloaders, num_classes):
    def client_fn(cid: str):
        return FlowerClient(vcid[int(cid)], trainloader=trainloaders[int(cid)], validationloader=validationloaders[int(cid)], num_classes=num_classes).to_client()
    
    return client_fn

def cli_eval_distr_results(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m['acc_distr'])
        vcid.append(m['cid'])

    # Aggregate and return custom metric (weighted average)
    return {"acc_distr": acc, "cid": vcid}

def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    vcid = []
    for num_examples, m in metrics:
        acc.append(m['acc_val_distr'])
        vcid.append(m['cid'])

    # Aggregate and return custom metric (weighted average)
    return {"acc_val_distr": acc, "cid": vcid}


#def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
#    # Multiply accuracy of each client by number of examples used
#    accuracies = [num_examples * m["acc_distr"] for num_examples, m in metrics]
#    examples = [num_examples for num_examples, _ in metrics]
#
#    # Aggregate and return custom metric (weighted average)
#    return {"acc_distr_weighted": sum(accuracies) / sum(examples)}
#

