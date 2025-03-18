# VANILLA CNL PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from dataset import prepare_dataset_cnl

import matplotlib.pyplot as plt
import numpy as np

import pickle
from pathlib import Path

import logging
import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from model import Net, LeNet, train, test#, init_normal


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    #1. LOAD CONFIGURATION
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # Load config params
    epochs = cfg.num_rounds
    lr = cfg.config_fit.lr
    batch_size = cfg.batch_size
    num_classes = cfg.num_clases

    #2. LOAD DATASET    
    trainloader, validationloader, testloader = prepare_dataset_cnl(batch_size=batch_size, seed=cfg.seed)

    #3. TRAINING
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    #model = Net(num_classes).to(device)
    model = LeNet(num_classes).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    #from tqdm import tqdm
    train_loss, metrics_val_distributed_fit = train(model, trainloader, validationloader, optim, epochs, num_classes, device)
    
    #4. VALIDATION
    history = []

    #5. EVALUATION
    loss, accuracy = test(model, testloader, num_classes, device)

 
    #6. SAVE RESULTS
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    out = "**accuracy: " + str(accuracy) + "\n**training_loss: " + ' '.join([str(elem) for elem in train_loss]) + '\n'
    f = open(save_path + "/output.out", "w")
    f.write(out)
    f.close()

if __name__ == "__main__":
    main()
