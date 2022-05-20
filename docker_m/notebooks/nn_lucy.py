from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.optim as optim


# from torch

# https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/



class nnModel(nn.Module): # Modelo Pensado para classificação
    def __init__(self, inFeatures=9,random_state=42):
        super(nnModel, self).__init__()

        self.inFeatures = inFeatures

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.define_arch()

        torch.manual_seed(random_state)


        self.criterion = nn.BCELoss()


    def define_arch(self):
        self.fc1 = nn.Linear(in_features=self.inFeatures, out_features=64)

        self.fc2 = nn.Linear(in_features=64, out_features=32)

        self.fc3 = nn.Linear(in_features=32, out_features=1)

        nnModel.init_weights(self.fc1)
        nnModel.init_weights(self.fc2)
        nnModel.init_weights(self.fc3)


    def init_weights(layer):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

    def fit(self, trainloader=[], validloader=[], valid_int=10, numEpochs=10, learningRate=0.1, momentum=0.0, gamma=1.0, regularization=0.0, verbose=False, device=None):
        if device == None:
            device = self.device

        # Definindo o modo de operação do modelo
        self.train()

        # Passando o modelo para CPU/GPU
        self = self.to(device)

        # Creterio
        # criterion = nn.MSELoss()
        # if self.regression:
        #     criterion = nn.L1Loss()
        # else:
        #     criterion = nn.BCELoss()
        # criterion = nn.BCEWithLogitsLoss()
        # https://medium.com/dejunhuang/learning-day-57-practical-5-loss-function-crossentropyloss-vs-bceloss-in-pytorch-softmax-vs-bd866c8a0d23
        # criterion = nn.CrossEntropyLoss()

        # Optimizador
        optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=momentum, weight_decay=regularization)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        losses = []
        accur = []

        for epoch in range(numEpochs):
            epoch_loss = 0
            for batch_id, (x_train, y_train) in enumerate(trainloader, 0):

                y_train = y_train.reshape(-1, 1)

                x_train, y_train = x_train.to(device), y_train.to(device)

                # Passando features pela rede
                pred_labels = self(x_train)
            
                loss = self.criterion(pred_labels, y_train)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            scheduler.step()

            if verbose:
                print("[{epc}/{epcs}] total epoch loss: {loss:.4f}".format(
                    epc=epoch+1, epcs=numEpochs, loss=epoch_loss))

            losses.append(epoch_loss)

            if(validloader != [] and ((epoch+1) % valid_int == 0)):
                valid_correct = 0
                valid_total = 0
                valid_total_loss = 0
                self.eval()
                with torch.no_grad(): 
                    for valid_batch_id, (x_valid, y_valid) in enumerate(validloader, 0):

                        y_valid = y_valid.reshape(-1, 1)
                        x_valid, y_valid = x_valid.to(device), y_valid.to(device)

                        pred_labels_valid = self(x_valid)

                        valid_loss = self.criterion(pred_labels_valid, y_valid)
                        valid_total_loss += valid_loss.item()

                        valid_total += pred_labels_valid.size(0)
                        valid_correct += (pred_labels_valid.cpu().reshape(-1).detach().numpy().round(
                        ) == np.array(y_valid.cpu().reshape(-1).detach().numpy().round(), int)).sum()

                valid_acc = 100*valid_correct/valid_total
                accur.append(valid_acc)
                if verbose:
                    print("(Validation Set) Epoch: {epc}, last batch loss: {l:.4f}, accuracy: {acc:.2f}".format(epc=epoch+1,l=loss.item(),acc=valid_acc))


                self.train()

        return losses, accur

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            predicted = self(features)

        return predicted.cpu().reshape(-1).detach().numpy().round()

class regNN_model(nnModel):
    def __init__(self, inFeatures=9, random_state=42):
        super(regNN_model,self).__init__(inFeatures, random_state)

        self.criterion = nn.L1Loss()

    def define_arch(self):
        self.fc1 = nn.Linear(in_features=self.inFeatures, out_features=64)

        self.fc2 = nn.Linear(in_features=64, out_features=32)

        self.fc3 = nn.Linear(in_features=32, out_features=1)

        nnModel.init_weights(self.fc1)
        nnModel.init_weights(self.fc2)
        nnModel.init_weights(self.fc3)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)

        # x = torch.round(x)

        return x


from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class dts(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def __len__(self):
    return self.x.shape[0]


def dataset_splitter(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

