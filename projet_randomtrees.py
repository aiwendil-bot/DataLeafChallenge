from sklearn import svm
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics

lab_enc = preprocessing.LabelEncoder() #empêche l'erreur ValueError: Unknown label type: 'continuous'


#import des données
leafs_train = np.loadtxt(r'C:\Users\Adrien\Documents\projet_challenge_donnees_adrien_callico\leaf_train_data.csv')

leafs_labels = np.loadtxt(r'C:\Users\Adrien\Documents\projet_challenge_donnees_adrien_callico\leaf_train_labels.csv')

leafs_prediction = np.loadtxt(r'C:\Users\Adrien\Documents\projet_challenge_donnees_adrien_callico\leaf_test_data.csv')

X = leafs_train
y = leafs_labels
y = y.astype('long')

#recentrage des données
X = preprocessing.scale(X)
y = preprocessing.scale(y)
leafs_prediction=preprocessing.scale(leafs_prediction)

#séparation en données d'entrainement et de test (70-30)

indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx, :]
y_train = y[training_idx]
X_test = X[test_idx, :]
y_test = y[test_idx]

y = lab_enc.fit_transform(y)
y_train = lab_enc.fit_transform(y_train)

#création du modèle
clf=RandomForestClassifier(n_estimators=10, max_depth=None,   min_samples_split=2, random_state=0).fit(X_train,y_train)

#prédiction
y_pred=clf.predict(X)

#affichage du taux de réussite
print("Accuracy:",metrics.accuracy_score(y, y_pred))

#si besoin, affichage de la prédiction (format liste)
#print(y_pred.tolist())


