import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets,preprocessing
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F


#import des données

leafs_train = np.loadtxt('leaf_train_data.csv')

leafs_labels = np.loadtxt('leaf_train_labels.csv')

leafs_prediction = np.loadtxt('leaf_test_data.csv')

leafs_prediction = preprocessing.scale(leafs_prediction)

X = leafs_train
#X=preprocessing.scale(X)
y = leafs_labels
#y=preprocessing.scale(y)

##précision /!\ je sais qu'il faut preprocess les données, mais cela me donne une erreur "IndexError: Target -1 is out of bounds." que je n'ai pas réussi à résoudre



#paramètres du modèle
d = X.shape[1]
k = 36

#fonction qui calcule les prédictions à partir des sorties du modèle
def prediction(f):
    return th.argmax(f, 1)

#fonction qui calcule le taux d'erreur en comparant les y prédits avec les y réels
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

#séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]

#création du modèle de régression logistique multivarié
class Reg_log_multi(th.nn.Module):

    def __init__(self,d,k):
        super(Reg_log_multi, self).__init__()

        self.layer = th.nn.Linear(d,k)
        self.layer.reset_parameters()

    def forward(self, x):
        out = self.layer(x)
        return th.softmax(out,1)


#creation d'un objet modele de régression logistique multivarié avec les paramètres d et k
model = Reg_log_multi(d,k)

#spécification du materiel utilisé device (cpu car pas de CG sur les pc de la fac
device = "cpu"

#passage du modèle sur le device
model = model.to(device)


#conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).to(device)
y_train = y_train.long() #passage des données en type 'long'
X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).to(device)
y_test = y_test.long()


#learning rate
eta = 0.01

#définition du critère de perte
criterion = th.nn.CrossEntropyLoss()

#correspond à la descente de gradient (Adam produit de bons résultats en général)
optimizer = optim.Adam(model.parameters(), lr=eta)

# tqdm permet d'avoir une barre de progression
nb_epochs = 100000
pbar = tqdm(range(nb_epochs))
#listes qui serviront à l'affichage de l'évolution du taux d'erreur
liste_error_test =[]
liste_error_train =[]
for i in pbar:
    # Remise à zéro des gradients
    optimizer.zero_grad()

    f_train = model(X_train)#si on souhaite s'entrainer sur toutes les données, on remplace X_train et y_train par X et y
    loss = criterion(f_train,y_train)
    # Calculs des gradients
    loss.backward()

    # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = model(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())

        liste_error_test.append(error_test)
        liste_error_train.append(error_train)

#prédiction du modele entrainé sur l'ensemble
z=model(th.from_numpy(leafs_train).float().to(device)) #tenseur de dim 240x36

yhat=th.max(z.data,1)#renvoie un tenseur de dim 1 qui contient l indice du max pour chaque ligne (donc la classe prédite)

print(yhat)

#affichage de l'évolution du taux d'erreur
X = [i for i in range(100)] #nb d'itérations /1000
Y1 = liste_error_test
Y2 = liste_error_train

plt.xlabel("nb itérations x1000")
plt.ylabel("taux erreur")


plt.plot(X,Y1, label="error_test")
plt.plot(X,Y2, label="error_train")
plt.legend()
plt.show()



