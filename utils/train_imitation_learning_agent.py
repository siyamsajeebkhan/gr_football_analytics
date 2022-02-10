import sys
sys.path.append("..")

import logging
import os
# import time
# import datetime
import random
import pandas as pd
import numpy as np
import torch
# import torch.nn.functional as F
import torch.nn as nn
# import torch.optim as optim
import csv
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
# from os.path import join, dirname
# from sklearn.model_selection import train_test_split
from imitation_learning.dataset.frame_dataset import Float115Dataset
from models.mlp import MLPModel
from utils.solver import Solver
# from gfootball.env.wrappers import Simple115StateWrapper
from solver import Solver
from torch.utils.tensorboard import SummaryWriter
# from imitation_learning.dataset.dict_dataset import Float115Dataset as DictDataset
# import hydra
# import wandb
# from omegaconf import DictConfig
# import torchtoolbox.transform as transforms
# import torchvision
# import pickle

def train_model():
    # Setting the seeds for result reproducibility
    os.environ['PYTHONHASHSEED'] = str(42)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Loading the dataset containing names of all the frames
    dataset = pd.read_csv('../data/frames.csv', header=None)[0]

    # Creating Train, Val and Test Dataset
    train, val, test = np.split(dataset.sample(frac=1, random_state=42), [
                                        int(.6 * len(dataset)), int(.8 * len(dataset))])

    train_frames, val_frames, test_frames = np.array(train, dtype='str'),\
                                            np.array(val, dtype='str'),\
                                            np.array(test, dtype='str')

    dataset_path = '/home/ssk/Study/GRP/dataset/npy_files'
    train_dataset, val_dataset, test_dataset = Float115Dataset(train_frames, dataset_path), \
                                                Float115Dataset(val_frames, dataset_path), \
                                                Float115Dataset(test_frames, dataset_path)

    logging.info("Number of training samples:", len(train_dataset))
    logging.info("Number of validation samples:", len(val_dataset))
    logging.info("Number of test samples:", len(test_dataset))    

    del(train)
    del(val)
    del(test)
    del(dataset)
    del(train_frames)
    del(val_frames)
    del(test_frames)

    # Creating the dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=4)

    lr = 1e-3

    # Loading the model and defining different parameters for the training
    model = MLPModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=1, verbose=True, factor=0.2)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    solver = Solver(model, train_loader, val_loader, criterion, lr, optimizer, writer=writer)
    solver.train(epochs=1)

if __name__ == '__main__':
    train_model()