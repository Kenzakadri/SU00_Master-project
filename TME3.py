import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import graphviz
import numpy as np
import matplotlib.pyplot as plt


print("TME3")
data_diabetes = pd.read_table('patients_data.txt',sep='\t',header=None)
classes_diabetes = pd.read_table('patients_classes.txt',sep='\t',header=None)

print("0. Classes")

print("Voici les classes réelles : ")
classe = classes_diabetes.values
classe.transpose()
classe = [i[0] for i in classe]
print(classe)

print("1. Arbre de décision")
dt = tree.DecisionTreeClassifier()
dt.fit(data_diabetes, classes_diabetes)


feature_names = ['age', 'hba1c', 'insuline taken', 'other drugs taken']
classes = ['DR','NDR’']
dot_data= tree.export_graphviz(dt, out_file=None,feature_names=feature_names,class_names=classes,filled=True, rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)

graph.render("diabetes remission")


dt = dt.predict(data_diabetes)
print("Arbre de décision = ",dt)
print("Moyenne = ", np.mean(dt))
print("Ecart-type = ", (np.std(dt)))



print("Comparaison des classes réelles et de l'arbre de décision")
data1 = dt-classe
print(data1)
print("On remarque que si on soustrait les deux matrices, il n'y a aucune différence. ")
a = 0
for i in data1:
    if i!=0 :
        a+=1

print("Pourcentage d'erreur = ", a/len(data1)*100)
print("Le pourcentage d'erreur est de 0, l'arbre de décision donne des valeurs identiques aux classes réelles. \n Ce qui s'explique par le fait que la base d'apprentissage de l'arbre de décision est la matrice entière des classes réelles. \n Donc on demande à cet arbre de décision les données sur lesquelles il a appris")

print("2. Forêt d'arbres décisionnels / Random forest")


rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(data_diabetes, classes_diabetes)

##################
importance = rf.feature_importances_
print(rf.feature_importances_)
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importance)[::-1]

# Print the feature ranking
print("Feature ranking:")
clinicalvariables = ("Age", "HbA1C", "Prise d'insuline ", "Autres traitements")
sortedclinicalvariables=[]

for f in range(data_diabetes.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
    sortedclinicalvariables.append(clinicalvariables[indices[f]])

# Plot the feature importances of the forest
plt.figure()
plt.title("Importances des variables ")
plt.bar(range(data_diabetes.shape[1]), importance[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(data_diabetes.shape[1]), sortedclinicalvariables)
plt.xlim([-1, data_diabetes.shape[1]])
plt.show()



rf = rf.predict(data_diabetes)

print("random forest", rf)
print("moyenne", np.mean(rf))
print("std",np.std(rf))

data2 = rf-classe
print("Comparaison des classes réelles avec la prediction du random forest")
print(data2)

a = 0
for i in data2:
    if i!=0 :
        a+=1

print("Pourcentage d'erreur",a/len(data2)*100)
print("On retrouve 23.5% d'erreur entre les données réelles et les prédictions du random tree. "
      "\n Contrairement à l'arbre de décision, cette fois le classifieur apprend sur des arbres de decision provenant de sous-ensembles de la bases d'apprentisssage \n Ce classifier repose sur le principe de bootsraping ce qui lui permet d'apprendre à partir des arbres de décision différents malgré une unique base d'apprentissage")


print('3. Diarem')


print(data_diabetes.iloc[:,0])
Diarem = np.zeros(len(data_diabetes.iloc[:, 0]))

for i in range(len(data_diabetes.iloc[:,0])):
    if data_diabetes.iloc[i,0] < 40:
        Diarem[i] = 0
    elif data_diabetes.iloc[i,0] >=40 and data_diabetes.iloc[i,0] <= 50:
        Diarem[i] = 1
    elif data_diabetes.iloc[i,0] >50 and  data_diabetes.iloc[i,0] <60:
        Diarem[i] = 2
    elif data_diabetes.iloc[i,0] >= 60:
        Diarem[i] = 3
for i in range(len(data_diabetes.iloc[:,1])):
    if data_diabetes.iloc[i,1] < 6.5:
        Diarem[i] += 0
    elif data_diabetes.iloc[i,1] >=6.5 and data_diabetes.iloc[i,1] <= 6.9:
        Diarem[i] += 2
    elif data_diabetes.iloc[i,1] >7 and  data_diabetes.iloc[i,1] <8.9:
        Diarem[i] += 4
    elif data_diabetes.iloc[i,1] >= 9:
        Diarem[i] += 6
for i in range(len(data_diabetes.iloc[:,2])):
    if data_diabetes.iloc[i,2] == 0:
        Diarem[i] += 0
    elif data_diabetes.iloc[i,2] == 1:
        Diarem[i] += 10
for i in range(len(data_diabetes.iloc[:,3])):
    if data_diabetes.iloc[i,3] == 0:
        Diarem[i] += 0
    elif data_diabetes.iloc[i,3] == 1:
        Diarem[i] += 3
for i in range(len(data_diabetes.iloc[:,3])):
    if Diarem[i] < 7 :
        Diarem[i] = 0
    elif Diarem[i] >=7:
        Diarem[i] = 1

print(Diarem)
print("Moyenne = ", np.mean(Diarem))
print("Ecart type = ", np.std(Diarem))

print("Comparaison des classes réelles avec le score Diarem")
data3 = Diarem-classe
print(data3)

a = 0
for i in data3:
    if i!=0 :
        a+=1

print("Pourcentage d'erreur = ",a/len(data3)*100)
print("Cette fois on retrouve un taux d'erreur de 27%. \n Ce score a été mis en place sur 690 patients du même hopital ce qui peut biaisé les résultats. \n De plus, un poids plus important a été donné à la prise d'insuline dans cette étude, contrairement au random forest qui retrouve un poids plus important pour l'âge. \n Le score Diarem a été mis au point en triant 250 variables pour ne garder que les 4 variables les plus influentes. \n Il est difficile de comparer des classifieurs se basant sur une base d'apprentissage avec 50% de rémission, et un score se basant sur une sorte de base d'apprentissage avec 63% de rémission (en moyenne on note 60% de rémission après la chirurgie)). \n Aussi avoir une plus grande base d'apprentissage aurait était d'autant plus intéressant pour comparer tous les classifieurs  ")

print("Conclusion : \n Donc pour conclure, l'arbre de décision n'est pas utile dans ce cas car la base d'apprentissage = les classes réelles, le random forest est intéressant, ainsi que le score Diarem, ce dernier \n a un plus grand pourcentage d'erreur mais il pourrait être intéressant de les comparer avec une même base d'apprentissage")