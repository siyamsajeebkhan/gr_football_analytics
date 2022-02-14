import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import pickle
import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader, random_split
from imitation_learning.dataset.frame_dataset import Float115DatasetBatched

class MLPModel(pl.LightningModule):
    """
    Take float115_v2 (115 dimension vector) as input
    """
    def __init__(self, hparams, input_size=115, hidden_size=256, num_classes=19):
        super().__init__()

        # set hyperparameters
        self.hparams.update(hparams)
        # self.hidden_size = hparams.get('hidden_size', 128)
#         self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        self.batch_size = hparams['batch_size']
        self.activation = hparams['activation']
        if self.activation == 'LeakyReLU':
            act_func = nn.LeakyReLU()
        elif self.activation == 'ReLU':
            act_func = nn.ReLU()
        else:
            act_func = nn.PReLU()
        # self.model = None 
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_func,
            nn.Linear(hidden_size, hidden_size),
            act_func,
        )
        self.actor = nn.Linear(hidden_size, num_classes)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.model(x)
        # logits = self.actor(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        obs, targets = batch
        obs, targets = obs.reshape((obs.shape[0]*obs.shape[1], obs.shape[2])),\
                            targets.reshape((-1,))
        # load X, y to device!
        obs, targets = obs.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(obs)

        # loss
        loss = F.cross_entropy(out, targets.long())

        # if batch_idx == 0:
        #     self.visualize_predictions(obs, out, targets)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {'loss': loss}
        return {'loss': loss, 'train_n_correct':n_correct.detach(), 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct':n_correct.detach()}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def visualize_predictions(self, obs, preds, targets):
        obs = obs.to('cpu')
        preds = preds.to('cpu')
        targets = targets.to('cpu')
        class_names = [x for x in range(19)]

        # determine size of the grid based on given batch size
        num_rows = torch.tensor(len(obs), device='cpu').float().sqrt().floor()

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(obs)):
            plt.subplot(num_rows, len(obs) // num_rows + 1, i+1)
            plt.imshow(obs[i].squeeze(0))
            plt.title(class_names[int(torch.argmax(preds, dim=-1)[i])] + f'\n[{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure('predictions', fig, global_step=self.global_step)

    # For training with top 480 replay files
    # def prepare_data(self):
    #     # download
    #     with open('../data/top_480_replay_data.pkl', 'rb') as pkl_file:
    #         dict_dataset = pickle.load(pkl_file)
    #         data = pd.DataFrame(dict_dataset.keys())

    #     # Creating Train, Val and Test Dataset
    #     torch.manual_seed(42)
    #     # train, val, test = random_split(data, [int(N*0.6), int(N*0.2), int(N*0.2)])
    #     train, val, test = np.split(data.sample(frac=1, random_state=42), [
    #                                 int(.6 * len(data)), int(.8 * len(data))])
    #     train, val, test = train.reset_index(drop=True), \
    #                                             val.reset_index(drop=True), \
    #                                             test.reset_index(drop=True)

    #     self.dataset = {}
    #     self.dataset["train"], self.dataset["val"], self.dataset["test"] = Float115Dataset(train, dict_dataset), \
    #         Float115Dataset(val, dict_dataset), \
    #             Float115Dataset(test, dict_dataset)
    #     torch.manual_seed(torch.initial_seed())
    #     del(dict_dataset)
    #     del(data)
    #     del(train)
    #     del(test)
    #     del(val)
    #     del(pkl_file)

    def prepare_data(self):

        # download
        root_dir = '/home/ssk/Study/GRP/dataset/batched_data'
        train_data_path = os.path.join(root_dir, 'train')
        val_data_path = os.path.join(root_dir, 'val')
        test_data_path = os.path.join(root_dir, 'test')

        # Creating Train, Val and Test Dataset
        torch.manual_seed(42)
        train, val, test = np.array(sorted(os.listdir(train_data_path)), dtype='str'),\
                    np.array(sorted(os.listdir(val_data_path)), dtype='str'),\
                        np.array(sorted(os.listdir(test_data_path)), dtype='str')
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = Float115DatasetBatched(train, train_data_path),\
                                                                            Float115DatasetBatched(val, val_data_path),\
                                                                            Float115DatasetBatched(test, test_data_path)
        torch.manual_seed(torch.initial_seed())
        print(f"Length of train dataset: {len(self.dataset['train'])}")
        print(f"Length of val dataset: {len(self.dataset['val'])}")
        print(f"Length of test dataset: {len(self.dataset['test'])}")
        del(train)
        del(test)
        del(val)


#     @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.batch_size, num_workers=8)

#     @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.batch_size, num_workers=8)

#     @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
#         lr_schedulers = {"scheduler": ReduceLROnPlateau(opt1, ...), "monitor": "metric_to_track"}
#         optim = ([optimizer],
#                 [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                             factor=self.hparams["lr_decay_rate"], patience=1)])
        optim = ([optimizer],
                [{"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=self.hparams["lr_decay_rate"], patience=1), "monitor": "metric_to_track"}])
#         {"scheduler": ReduceLROnPlateau(opt1, ...), "monitor": "metric_to_track"}
        return optim


    def getTestAcc(self, loader = None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X, y = X.reshape((X.shape[0]*X.shape[1], X.shape[2])),\
                    y.reshape((-1,))
            X, y = X.to(self.device), y.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc