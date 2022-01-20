import pandas as pd
import numpy as np
from os.path import isfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from d2_acc_dataset import Dota2Dataset
from torch.utils.data import DataLoader

#Dota 2 Kaggle Dataset - Match.csv and Players.csv
#Prediction will be based on hero selection *only*

#Kaggle Data
matches = pd.read_csv('./match.csv',usecols=['match_id','radiant_win'])
players = pd.read_csv('./players.csv',usecols=['match_id','hero_id','player_slot'])
heroes = pd.read_csv('./hero_names.csv')
print(matches.head())

#Map hero names to their ids
hero_dict = dict(zip(heroes['hero_id'],heroes['localized_name']))
hero_dict[0] = 'N/A'
players['hero'] = players['hero_id'].apply(lambda _id: hero_dict[_id])

#Merge matches with players
data = pd.merge(matches, players, on = 'match_id', how='left')
print("Match data merged with players:")
print(data.iloc[0:17])
#Encode the nominal features
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
    #Make a dataframe for each team's heroes
    radiant_heroes = pd.DataFrame(radiant_heroes, columns=radiant_cols)
    dire_heroes = pd.DataFrame(dire_heroes,columns=dire_cols)
    #Concatenate the two dataframes
    X = pd.concat([radiant_heroes, dire_heroes], axis=1)
    #Output this to a .csv file to reuse later, since this is a costly operation
    X.to_csv('match_heroes.csv', index=False) 
    
print("\nHead() of encoded hero win/loss dataset: \n", X.head())

#Need to encode our labels
y = matches['radiant_win'].apply(lambda win: 1 if win else 0)

#Logistic Regression Classifier Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
# LR Pipeline with Standard Scaler
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
#LR stats
print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('CM:', confusion_matrix(y_test, y_pred))
print('Report:', classification_report(y_test, y_pred))
print('Cross Validation', cross_val_score(pipe_lr, X, y, cv=5))

