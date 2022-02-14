import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import time

import torch.nn.functional as F

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import random
import os

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    or regression models.
    The Solver performs gradient descent using the given learning rate.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, learning_rate to the constructor.
    You will then call the train() method to run the optimization
    procedure and train the model.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_loss_history and solver.val_loss_history will be lists containing
    the losses of the model on the training and validation set at each epoch.
    """

    def __init__(self, model, train_dataloader, val_dataloader,
                 learning_rate, optimizer, verbose=True, print_every=1, writer=None):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above

        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data

        - loss_func: Loss function object.
        - learning_rate: Float, learning rate used for gradient descent.

        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.learning_rate = learning_rate
#         self.loss_func = loss_func
        self.writer = writer
        self.opt = optimizer
        # print(f"Optimizer: {optimizer}")
        # if optim == "Adam":
        #     print("True")
        #     self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.verbose = verbose
        self.print_every = print_every

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.best_val_loss = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_batch_loss = []
        self.val_batch_loss = []

        self.num_operation = 0

    def train(self, epochs=100, checkpoint_dir = None):
        """
        Run optimization to train the model.
        """
        # Start an epoch
        for epoch in range(epochs):
            print("Running the training loop")
            # running_loss = 0.0
            start_time = time.time()
            epoch_steps = 0
            # Iterate over all training samples
            train_epoch_loss = 0.0
            self.model.train()
            for i, data in tqdm(enumerate(self.train_dataloader, 0)):
                # Unpack data
                inputs, labels = data
                inputs, labels = inputs.reshape((inputs.shape[0]*inputs.shape[1], inputs.shape[2])),\
                                    labels.reshape((-1,))
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                # X = torch.as_tensor(X, device='cuda', dtype=torch.float32)
                # y = torch.as_tensor(y, device='cuda', dtype=torch.float32)


                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                train_epoch_loss += loss.item()
                epoch_steps += 1
                if i % 10000 == 9999:  # print every 10000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0

            # Validation loss
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(self.val_dataloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.reshape((inputs.shape[0]*inputs.shape[1], inputs.shape[2])),\
                                        labels.reshape((-1,))
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = F.cross_entropy(outputs, labels.long())
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            # with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #     path = os.path.join(checkpoint_dir, "checkpoint")
            #     torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)

            # tune.report(loss=(val_loss / val_steps), accuracy=correct / total, train_epoch_loss=(train_epoch_loss / epoch_steps))
            if self.verbose and epoch % self.print_every == 0:
                print(f"Epoch {t + 1} / {epochs})| train loss: {(train_epoch_loss / epoch_steps)} | val loss: {(val_loss / val_steps)} | epoch time: {str(datetime.timedelta(seconds=time.time() - start_time))[:7]}")
        
            # Keep track of the best model
            self.update_best_loss(val_loss / val_steps)

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        print("Finished Training")

    def get_dataset_prediction(self, loader):
        prediction = []
        for batch in loader:
            X = batch['features']
            y_pred = self.model.forward(X)
            prediction.append(y_pred)
        return np.concatenate(prediction, axis=0)

    def update_best_loss(self, val_loss):
        PATH = '../saved_models/best_mlp_model.pth'
        
        # Update the model and best loss if we see improvements.
        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_params = self.model.state_dict()
            torch.save(self.best_params, PATH)