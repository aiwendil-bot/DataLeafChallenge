import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
from sklearn import preprocessing


#data
leafs_train = np.loadtxt('leaf_train_data.csv')

leafs_labels = np.loadtxt('leaf_train_labels.csv')

leafs_prediction = np.loadtxt('leaf_test_data.csv')

leafs_prediction = preprocessing.scale(leafs_prediction)

X = leafs_train
X = preprocessing.scale(X)


y = leafs_labels
y = y.astype('long')


d = X.shape[1]
k = 36

#prediction
def prediction(f):
    return th.argmax(f, 1)

#taux d erreur
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]



# Séparation des données en un ensemble d'apprentissage (70%) et de test (30%)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]

# réseau de neurones

class Neural_network_binary_classif(th.nn.Module):

    def __init__(self,d,k,h1,h2):
        super(Neural_network_binary_classif, self).__init__()

        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()

    def forward(self, x):
        phi1 = th.sigmoid(self.layer1(x))
        phi2 = th.sigmoid(self.layer2(phi1))

        return th.softmax(self.layer3(phi2),1)


# instanciation du réseau de neurones

nnet = Neural_network_binary_classif(d,k,25,25)

#cpu ou CG si disponible
device = "cpu"

nnet = nnet.to(device)


# Conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).to(device)
y_train = y_train.long()
X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).to(device)
y_test = y_test.long()

# Taux d'apprentissage
eta = 0.01

#critère de perte
criterion = th.nn.CrossEntropyLoss()

# descente de gradient
optimizer = optim.Adam(nnet.parameters(), lr=eta)

#barre de progression
nb_epochs = 100000
pbar = tqdm(range(nb_epochs))

liste_error_test =[]
liste_error_train =[]

for i in pbar:
    optimizer.zero_grad()

    f_train = nnet(X_train)

    loss = criterion(f_train,y_train)

    loss.backward()

    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = nnet(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)
        liste_error_test.append(error_test)
        liste_error_train.append(error_train)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())

#test du modèle entrainé sur leaf_train_data.csv
z=nnet(th.from_numpy(X).float().to(device)) #tenseur de dim 240x36
yhat=th.max(z.data,1) #renvoie un tenseur de dim 1 qui contient l indice du max pour chaque ligne (donc la classe prédite)
print(yhat)

#affichage de l'évolution du taux d'erreur

X = [i for i in range(100)] #nb d'itérations /1000
Y1 = liste_error_test
Y2 = liste_error_train

plt.xlabel("nb itérations x1000")
plt.ylabel("taux erreur")
plt.plot(X,Y2,label='error_test')
plt.plot(X,Y2, label="error_train")
plt.legend()
plt.show()
































