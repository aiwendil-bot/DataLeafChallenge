## affichage du graphique page 5 du rapport

import matplotlib.pyplot as plt

X = [5,10,20,30,50]
Y1 = [83,86.4,90.3,90.24,90]
Y2 = [87.04,88.75,89.7,91.22,92]

plt.scatter(X,Y1, color = "blue",label="RandomForestClassifier")
plt.scatter(X,Y2,color="red",label="ExtraTreesClassifier")
plt.xlabel("Nombre d'arbres générés")
plt.ylabel("% de réussite")
plt.title("Comparaison entre RandomForestClassifier et ExtraTreesClassifier")
plt.legend()
plt.show()