
## ce petit script me permettait de vérifier rapidement le taux de bonnes prédictions de mes modèles sur les données d entrainement, et si besoin de l'enregistrer au format csv.

import numpy as np

array_prediction = [12, 14, 13, 9, 24, 17, 23, 9, 3, 0, 1, 18, 19, 17, 4, 4, 13, 6, 11, 10, 0, 7, 11, 14, 19, 11, 19, 3, 12, 0, 22, 21, 4, 17, 17, 27, 13, 5, 12, 19, 19, 9, 0, 4, 1, 8, 17, 27, 15, 15, 7, 7, 3, 13, 21, 17, 29, 15, 10, 20, 6, 9, 10, 28, 0, 14, 12, 20, 1, 6, 9, 10, 2, 7, 8, 0, 28, 22, 27, 23, 26, 4, 9, 14, 7, 13, 8, 1, 9, 6, 10, 4, 23, 2, 16, 6, 5, 26, 10, 17, 25, 4, 0, 1, 8, 15, 18, 12, 8, 11, 12, 4, 10, 11, 16, 22, 8, 29, 28, 23, 0, 1, 5, 4, 20, 13, 8, 14, 22, 26, 20, 29, 5, 22, 20, 11, 29, 24, 10, 10, 27, 28, 25, 8, 13, 25, 23, 10, 27, 24, 10, 6, 28, 12, 19, 16, 13, 19, 5, 1, 2, 12, 2, 5, 10, 1, 27, 15, 28, 7, 18, 17, 23, 25, 10, 0, 13, 10, 22, 29, 7, 2, 1, 25, 27, 0, 9, 6, 21, 29, 28, 2, 1, 3, 12, 29, 12, 7, 8, 8, 1, 23, 15, 14, 15, 11, 28, 23, 12, 27, 28, 25, 8, 28, 7, 1, 9, 18, 15, 26, 24, 17, 11, 1, 3, 15, 15, 3, 23, 18, 19, 17, 7, 28, 14, 19, 22, 18, 7, 11]

labels = np.loadtxt('leaf_train_labels.csv')

egalite = []

for i in range(len(array_prediction)):
    if (array_prediction[i]==labels[i]):
        egalite.append(1)
    else:
        egalite.append(0)

pourcentage = sum(egalite)/len(array_prediction) *100
print(pourcentage)

#le cas échéant, enregistre ma prédiction au format CSV pour l'envoyer sur le site
# np.savetxt(r"C:\Users\Adrien\Documents\projet\test_prediction.csv",array_prediction, delimiter=';')