from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch.autograd import Variable
from data_logreg import X, y
from d2-dataset import Dota2Dataset
import tensorboard as tb

#Data preprocessing for DataLoader, using a barebones custom Dataset class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

X_train = torch.Tensor(X_train.values).float()
y_train = torch.Tensor(y_train.values).float()
X_test = torch.Tensor(X_test.values).float()
y_test = torch.Tensor(y_test.values).float()

train_dataset = Dota2Dataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64)

test_dataset = Dota2Dataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=64)


class NeuralNet(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        #First layer has to be 222
        self.layer1 = nn.Linear(222, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.layer1(x)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        x = F.sigmoid(x)
        return x.squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = nn.BCELoss()(z, y)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(z, y))
        return loss

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        z = self.forward(x)
        loss = nn.BCELoss()(z, y)
        self.log('test_loss', loss, logger=True)
        return loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('test_loss', avg_loss, logger=True)
        self.log('test_acc', self.accuracy.compute(), logger=True)

    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.accuracy.compute(), prog_bar=True)

    def configure_optimizers(self, ):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

model = NeuralNet(X_train.shape[1], 2)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_dataloader, test_dataloader)
trainer.test(model, train_dataloader, test_dataloader)

#Save the model for use later
import pickle
d2model_save = model
pickle.dump(d2model_save, open("d2-model-save.p", "wb"))
