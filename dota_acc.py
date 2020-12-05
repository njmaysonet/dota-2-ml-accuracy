import pandas as pd
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

#Dota 2 Kaggle Dataset - Match.csv and Players.csv
#Prediction will be based on hero selection *only*

#Kaggle Data
matches = pd.read_csv('./dota2kaggle/match.csv',usecols=['match_id','radiant_win'])
players = pd.read_csv('./dota2kaggle/players.csv',usecols=['match_id','hero_id','player_slot'])
heroes = pd.read_csv('./dota2kaggle/hero_names.csv')

#Map hero names to their ids
hero_dict = dict(zip(heroes['hero_id'],heroes['localized_name']))
hero_dict[0] = 'N/A'
players['hero'] = players['hero_id'].apply(lambda _id: hero_dict[_id])

#Merge matches with players
data = pd.merge(matches, players, on = 'match_id', how='left')
print("Match data merged with players:")
print(data.iloc[0:17])

player_heroes = pd.get_dummies(players['hero'])
radiant_cols = list(map(lambda s: 'radiant_' + s, player_heroes.columns.values))
dire_cols = list(map(lambda s: 'dire_' + s, player_heroes.columns.values))

#We will encode the heroes present per match with 1 or 0. Each hero has two columns, one if they were on Radiant, one if they were on Dire
X = None

#It takes very long to map matches and heroes, so on subsequent runs, we'll reuse the mapping.
if isfile('match_heroes.csv'):
    X = pd.read_csv('match_heroes.csv')
else:

    radiant_heroes = []
    dire_heroes = []

    #So that we have one row per match, we will aggregate the hero counts
    for _id, _index in players.groupby('match_id').groups.items():
        radiant_heroes.append(player_heroes.iloc[_index][:5].sum().values)
        dire_heroes.append(player_heroes.iloc[_index][5:].sum().values)

    radiant_heroes = pd.DataFrame(radiant_heroes, columns=radiant_cols)
    dire_heroes = pd.DataFrame(dire_heroes,columns=dire_cols)
    X = pd.concat([radiant_heroes, dire_heroes], axis=1)
    X.to_csv('match_heroes.csv', index=False) 
print("Head() of encoded hero win/loss dataset: \n", X.head())

#Need to encode our labels
y = matches['radiant_win'].apply(lambda win: 1 if win else 0)
#classes = ['Dire Win', 'Radiant Win']

#Logistic Regression Classifier Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# LR ------------------------------------------------------------------------------------------
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('CM:', confusion_matrix(y_test, y_pred))
print('Report:', classification_report(y_test, y_pred))
print('Cross Validation', cross_val_score(pipe_lr, X, y, cv=5))

# PyTorch Lightning Neural Network
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class LinearPL(pl.LightningModule):
    def __init__(self):
        super(LinearPL, self).__init__()
        self.layer1 = nn.Linear(222,111)
        self.layer2 = nn.Linear(111,111)
        self.out = nn.Linear(111,1)
        self.relu = nn.ReLU()
        self.accuracy = pl.metrics.Accuracy()
        self.criterion = nn.BCELoss()
        self.batch_size = 222

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def get_weights(self):
        return self.weight
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.sigmoid(self.out(x))
        return x

    def training_step (self,batch,batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss, logger=True)
        self.log('train_step_acc', self.accuracy(y_hat, y))
        return loss
    
    def training_epoch_end(self, outputs):
        self.log('train_acc_epoch', self.accuracy.compute(), logger=True)

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        val_loss= self.criterion(y_hat, y.unsqueeze(1))
        self.log('val_loss', val_loss)
        return val_loss

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss', avg_loss, logger=True)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y.unsqueeze(1))
        self.log('test_loss', test_loss, logger=True)
        return test_loss

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('test_loss', avg_loss, logger=True)
        self.log('test_acc', self.accuracy.compute(), logger=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(Dota2Dataset(torch.Tensor(X_train.values), torch.Tensor(y_train.values)), batch_size=self.batch_size)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(Dota2Dataset(torch.Tensor(X_train.values), torch.Tensor(y_train.values)), batch_size=self.batch_size)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(Dota2Dataset(torch.Tensor(X_test.values), torch.Tensor(y_test.values)), batch_size=self.batch_size)

dota2model = LinearPL()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(dota2model)
trainer.test(dota2model)

import pickle
d2model_save = dota2model
pickle.dump(d2model_save, open("d2-model-save.p","wb"))
