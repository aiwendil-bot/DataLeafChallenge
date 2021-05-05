import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#import des données
leafs_train = np.loadtxt(r'C:\Users\Adrien\Documents\projet\leaf_train_data.csv')

leafs_labels = np.loadtxt(r'C:\Users\Adrien\Documents\projet\leaf_train_labels.csv')

leafs_prediction = np.loadtxt(r'C:\Users\Adrien\Documents\projet\leaf_test_data.csv')

X = leafs_train
y = leafs_labels

X = preprocessing.scale(X)

#split en données d'entrainement et données de test
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]

#mise sous un format reconnu par la classe xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 3,  # profondeur max d'un arbre
    'eta': 0.1,  # learning rate
    'silent': 1,# logging mode - quiet
    'objective': 'multi:softmax',  # evaluation de l'erreur
    'num_class': 36}  # nombre de classes à prédire
num_round = 10000  # nombre d'itérations
#entrainement
bst = xgb.train(param, dtrain, num_round)
#prédiction
preds = bst.predict(dtest)
#récupération de l'indice avec la plus haute probabilité pour chaque ligne, donc la classe prédite
best_preds = np.asarray([np.argmax(line) for line in preds])
#affichage du % de réussite
print ("% de réussite:", precision_score(y_test, best_preds, average='macro'))

# bst = xgb.train(param, dtrain, num_round)
# preds = bst.predict(dtest)
#
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print (best_preds.tolist())