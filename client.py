from collections import OrderedDict
from typing import Dict, Tuple, List
from flwr.common import NDArrays, Scalar

import json
import torch
import numpy as np
import flwr as fl
from model import LeNet, train, test, compute_prob_matrix#evaluate_loss_acc

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, validationloader, num_classes, seed, device):
        np.random.seed(seed)
        torch.manual_seed(seed)

        super().__init__()

        self.cid = cid
        self.validation_loaders = validationloader
        self.trainloader = trainloader[cid]
        self.validationloader = validationloader[cid]
        self.local_acc = None
        self.model = LeNet(num_classes)
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() and (device == 'GPU' or device == 'H100') else "cpu")
        #self.device = torch.device("mps")

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
   
    def fit(self, parameters, config):

        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)

        # Build class_client_matrix dynamically for this client
        val_counts = [0] * self.num_classes
        for _, labels in self.validation_loaders[self.cid]:
            for c in range(self.num_classes):
                val_counts[c] += (labels == c).sum().item()
        class_client_matrix = {self.cid: val_counts}

        # Determine number of epochs
        if int(config['comm_round']) <= 7:
            epochs = 15
            print(f" -> Client {self.cid}: Warm-up Phase Active! Training for {epochs} epochs.")
        elif config['nature'] == 'malicious':
            epochs = config['local_epochs']
        else:
            epochs = config['local_epochs']

        # All clients (head and neighbours) perform local training
        lr = config['lr']
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        #print("======TRAINING==== cid:", self.cid, "====head_cid:", config['head_cid'])
        _, metrics_val_distr, centroid = train(
            self.model, self.trainloader, self.validationloader,
            optim, epochs, self.num_classes, config['nature'], self.device
        )
        #centroid = [centroid]

        # Prob matrix always evaluated on head validation set
        prob_matrix = compute_prob_matrix(
            self.model, self.validation_loaders[config['head_cid']],
            self.num_classes, config['nature'], self.device
        )

        # Head client branch
        if config['head_cid'] == self.cid:
            print("======HERE==== HEAD", config['head_cid'], "====cid===", self.cid)
            return self.get_parameters({}), len(self.trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'centroid': [centroid],
                'confidence_score': [centroid],
                'prob_matrix': prob_matrix,
                'HEAD': 'YES',
                'distr_val_loss': '##',
                'energy used': '10W'
            }

        # Neighbour clients branch
        elif self.cid in config['neighbors']:
            # Step 1: neighbour's own centroid
            _, _, neighbour_centroid = test(
                self.model, self.validation_loaders[self.cid],
                self.num_classes, config['nature'], self.device
            )

            # Step 2: head's centroid
            _, _, head_centroid = test(
                self.model, self.validation_loaders[config['head_cid']],
                self.num_classes, config['nature'], self.device
            )

            # Debug evaluation
            #loss0, acc0 = evaluate_loss_acc(self.model, self.validation_loaders[self.cid], self.device)
            #print("client=>", self.cid, "=>Test-Neighbour-set=", neighbour_centroid,"===>Eval=>", acc0, "==>val_count=", class_client_matrix, "Head-set:", head_centroid)

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

            #print(f"Client=>{self.cid} => Conf-bf=>{neighbour_centroid} ====> conf-af===>{confidence_score}")
            #for pos, before, after in diffs:
            #    print(f"Cid=>{self.cid}=>Difference at position {pos}: neighbour={before}, confidence={after}")


            # Return neighbour metrics
            return self.get_parameters({}), len(self.trainloader), {
                'acc_val_distr': metrics_val_distr,
                'cid': self.cid,
                'centroid': [],
                'confidence_score': confidence_score,
                'prob_matrix': prob_matrix,
                'HEAD': 'NO',
                'distr_val_loss': '##',
                'energy used': '10W'
            }


    #############################################################################################
    def fit_aitor(self, parameters, config):

        #config['nature'], should be an array of neighbors
        torch.manual_seed(config['seed'])

        #copy params from server in local models
        self.set_parameters(parameters)
        metrics_val_distr = None
        centroid = None
        metrics_val_distr = 0.

        centroid_vector = []
        
        #Perform local training just in the selected node head
        if config['head_cid'] == self.cid:
            lr = config['lr']
            if config['comm_round'] <= config['num_agents']: # In first n initial rounds
                epochs = config['local_epochs'] # Option to achieve a faster converge in the first * 3 epochs
            elif config['nature'] == 'malicious':
                epochs = config['local_epochs']#*5
            else:
                epochs = config['local_epochs']

            optim = torch.optim.Adam(self.model.parameters(), lr=lr)
            #local training
            _, metrics_val_distr, centroid = train(self.model, self.trainloader, self.validationloader, optim, epochs, self.num_classes, config['nature'], self.device)
            centroid_vector.append(centroid)
            prob_matrix = compute_prob_matrix(self.model, self.validation_loaders[config['head_cid']], self.num_classes, config['nature'], self.device) # To compute centroids using neigh params in head val-set
            
            for neighbor in config['neighbors']: #USE MY PARAMS TO COMPUTE IN OTHER NEIGH VAL SETS -> CONFIDENCE
                if neighbor != self.cid:
                    _, _, centroid = test(self.model, self.validation_loaders[neighbor], self.num_classes, config['nature'], self.device)
                    centroid_vector.append(centroid)
            return self.get_parameters({}), len(self.trainloader), {'acc_val_distr': metrics_val_distr,'cid': self.cid, 'centroid': centroid_vector, 'prob_matrix': prob_matrix, 'HEAD': 'YES', 'distr_val_loss': '##', 'energy used': '10W'}
        elif self.cid in config['neighbors']:
            _, _, centroid = test(self.model, self.validation_loaders[config['head_cid']], self.num_classes, config['nature'], self.device) # To compute centroids using neigh params in head val-set
            prob_matrix = compute_prob_matrix(self.model, self.validation_loaders[config['head_cid']], self.num_classes, config['nature'], self.device) # To compute centroids using neigh params in head val-set
            #print('MC')
            #print(prob_matrix)
            _, metrics_val_distr, _ = test(self.model, self.validation_loaders[self.cid], self.num_classes, config['nature'], self.device) #To compute SCR / APPR1 with independent validation-sets in neighbor val-sets

        #Return current acc and params from neighbours
        return self.get_parameters({}), len(self.trainloader), {'acc_val_distr': metrics_val_distr,'cid': self.cid, 'centroid': centroid, 'prob_matrix': prob_matrix, 'HEAD': 'NO', 'distr_val_loss': '##', 'energy used': '10W'}

    #Evaluate global model in validation set of a particular client
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        #HERE I SHOULD COMPUTE OTHER CLIENTS PERFORMANCE
        torch.manual_seed(config['seed'])
        self.set_parameters(parameters)
        loss, accuracy, _ = test(self.model, self.validationloader, self.num_classes, config['nature'], self.device)
        return float(loss), len(self.validationloader), {'acc_distr': accuracy, 'cid': self.cid} #send anything, time it took to evaluation, memory usage...

def generate_client_fn(cids, trainloaders, validationloaders, num_classes, seed, device):
    def client_fn(cid: str):
        return FlowerClient(cids[int(cid)], trainloader=trainloaders, validationloader=validationloaders, num_classes=num_classes, seed=seed, device=device).to_client()
    return client_fn

def cli_eval_distr_results(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
    acc = []
    cids = []
    for num_examples, m in metrics:
        acc.append(m['acc_distr'])
        cids.append(m['cid'])
    # Aggregate and return custom metric (weighted average)
    return {"acc_distr": acc, "cid": cids}
####Ferdinand
def cli_val_distr_ferdos_to_review(metrics_list):
    confidence_all = []
    prob_matrix_all = []
    for m in metrics_list:
        # Collect confidence vectors
        if 'confidence' in m:
            confidence_all.append(m['confidence'])
        else:
            confidence_all.append([0.5] * 10)  # fallback

        # Collect prob_matrices
        if 'prob_matrix' in m:
            prob_matrix_all.append(m['prob_matrix'])
        else:
            prob_matrix_all.append(np.zeros((10,)))
    return confidence_all, prob_matrix_all

def cli_val_distr(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, List]:
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