import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

simplefilter(action='ignore', category=FutureWarning)
# Leer csv bank marketing
path = './dataSets/bank-marketing/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
columna age
'''
rangos = [20, 30, 40, 50, 60, 70, 80, 90]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
'''
Columna job
management 0
technician 1
entrepreneur 2
blue-collar 3
unknown 4
retired 5
'''
data.job.replace(['admin.','unknown','unemployed',
'management','housemaid','entrepreneur',
'student', 'blue-collar','self-employed',
'retired','technician','services'
], [0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
'''
Columna marital
married 0
single 1
divorced 2
'''
data.marital.replace(['married', 'single','divorced'], 
[0,1,2], inplace=True)
'''
Columna education
unknown 0
secondary 1
primary 2
tertiary 3
'''
data.education.replace(['unknown','secondary','primary',
'tertiary'], [0,1,2,3], inplace=True)
'''
Columna default
yes 0
no 1
'''
data.default.replace(['yes','no'], [0,1], inplace=True)
'''
Columna housing
yes 0
no 1
'''
data.housing.replace(['yes','no'], [0,1], inplace=True)
'''
Columna loan
yes 0
no 1
'''
data.loan.replace(['yes','no'], [0,1], inplace=True)
'''
columna balance,contact,day,month,
duration,pdays,previous,campaign(drop)
'''
data.drop(['balance','contact', 'day', 'month',
 'duration','pdays','previous','campaign'], axis= 1, inplace = True)
'''
Columna poutcome
unknown 0
other 1
failure 2
success 3
'''
data.poutcome.replace(['unknown', 'other','failure',
'success'], [0,1,2,3], inplace=True)
'''
Columna y
yes 0
no 1
'''
data.y.replace(['yes','no'], [0,1], inplace=True)
'''
NaN
'''
data.dropna(axis=0,how='any', inplace=True)
#Partir la tabla en dos
data_train = data[:22605]
data_test = data[22605:]
x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA
kfold = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

# MÉTRICAS

print('*'*50)
print('Regresión Logística Validación cruzada')
# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')


'''
# MAQUINA DE SOPORTE VECTORIAL CON VALIDACIÓN CRUZADA
svc = SVC(gamma='auto')
for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial Validación cruzada')
# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')


# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')
'''










