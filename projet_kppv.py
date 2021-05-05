import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import preprocessing

#import des données
leafs_train = np.loadtxt('leaf_train_data.csv')

leafs_labels = np.loadtxt('leaf_train_labels.csv')

leafs_prediction = np.loadtxt('leaf_test_data.csv')

#définition de la distance euclidienne

def euclidian_distance(v1,v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


#la fonction neighbors prend en paramètres le dataframe, une à tester et une valeur k et renvoie les k plus proches voisins de l’individu de test (selon la distance euclidienne)
def neighbors(X_train, y_label, x_test, k):

    list_distances =  []
    for i in range(X_train.shape[0]):

        distance = euclidian_distance(X_train[i,:], x_test)

        list_distances.append(distance)

    df = pd.DataFrame()

    df["label"] = y_label
    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k,:]

#la fonction prediction à partir d’un voisinage renvoie la classe majoritaire
def prediction(neighbors):
    mean = neighbors["label"].mode()[0]
    return mean

pred = prediction(nearest_neighbors)

# permet de faire une prédiction sur les données de test et de l'enregistrer sous format CSV pour l'envoyer sur codalabs
list_prediction = []
for i in range(leafs_prediction.shape[0]):
    list_prediction.append(prediction(neighbors(leafs_train, leafs_labels, leafs_prediction[i],1)))

array_prediction = np.array(list_prediction)

np.savetxt(r"C:\Users\Adrien\Downloads\projet\test_prediction_kpp.csv",array_prediction, delimiter=';')



#Echantillonage à 70% entre feuilles d entrainement et feuilles de test

indices = np.random.permutation(leafs_train.shape[0])

training_idx, test_idx = indices[:int(leafs_train.shape[0]*0.7)], indices[int(leafs_train.shape[0]*0.7):]

X_train = leafs_train[training_idx,:]
y_train = leafs_labels[training_idx]

X_test = leafs_train[test_idx,:]
y_test = leafs_labels[test_idx]

#le preprocessing permet de recentrer les données

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


cette fonction renvoie le taux de réussite sur les données d entrainement pour k voisins
def reussite_pour_k_voisins(k):
    reussite=0
    for i in range(X_test.shape[0]):
        nearest_neighbors = neighbors(X_train, y_train, X_test[i], k)
        if (prediction(nearest_neighbors) == y_test[i]):
            reussite+=1
    return (reussite / X_test.shape[0] * 100)


def liste_des_différentes_réussites():
    L =[]
    for k in range(1,15):
        mean = 0
        for i in range(100):
            mean += reussite_pour_k_voisins(k)
        L.append(mean/100)
    return L


#affichage
X = [i for i in range(1,15)]
Y = liste_des_différentes_réussites()

plt.xlabel("k plus proches voisins")
plt.ylabel("moyenne du % de réussite sur 100 simulations")
plt.xlim(0, 16)

plt.scatter(X,Y)
plt.show()










































