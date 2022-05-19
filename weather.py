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
# Leer csv weather
path = './dataSets/weather/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
Drops de columnas
'''
data.drop(['Date','Location','Rainfall','Evaporation','Sunshine',
'WindGustDir','WindDir9am', 
'WindDir3pm','RISK_MM'], axis= 1, inplace = True)
'''
columna Min Temperatura
'''
rangos = [ -8, 0, 10, 20, 35]
nombres = ['1', '2', '3', '4']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)
'''
columna Max Temperatura
'''
rangos = [ -5, 10, 20, 30, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.MaxTemp = pd.cut(data.MaxTemp, rangos, labels=nombres)
'''
columna WinGustSpeed
'''
data.WindGustSpeed.replace(np.nan, 39, inplace=True)
'''
columna WindSpeed9am
'''
rangos = [ 1, 26, 52 ,78, 94, 110, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos, labels=nombres)
'''
columna WindSpeed3pm
'''
rangos = [ 1, 17, 34, 52, 69, 87]
nombres = ['1', '2', '3', '4', '5']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos, labels=nombres)
'''
columna Humidity9am 
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity9am = pd.cut(data.Humidity9am, rangos, labels=nombres)
'''
columna Humidity3pm
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos, labels=nombres)
'''
columna Pressure9am
'''
rangos = [ 980, 994, 1008, 1022, 1036, 1050]
nombres = ['1', '2', '3', '4', '5']
data.Pressure9am = pd.cut(data.Pressure9am, rangos, labels=nombres)
'''
columna Pressure3pm
'''
rangos = [ 970, 984, 998, 1012, 1026, 1040]
nombres = ['1', '2', '3', '4', '5']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos, labels=nombres)
'''
columna Cloud9am
'''
data.Cloud9am.replace(np.nan, 4, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud9am = pd.cut(data.Cloud9am, rangos, labels=nombres)
'''
columna Cloud3pm
'''
data.Cloud3pm.replace(np.nan, 5, inplace=True)
rangos = [ 0, 1, 2, 3, 4, 5, 6, 7, 9]
nombres = ['1', '2', '3', '4', '5', '6', '7', '8']
data.Cloud3pm = pd.cut(data.Cloud3pm, rangos, labels=nombres)

'''
columna Temp9am
'''
rangos = [ -8, 0, 10, 20, 30, 42]
nombres = ['1', '2', '3', '4', '5']
data.Temp9am = pd.cut(data.Temp9am, rangos, labels=nombres)
'''
columna Temp3pm
'''
rangos = [ -6, 5, 15, 25, 40, 50]
nombres = ['1', '2', '3', '4', '5']
data.Temp3pm = pd.cut(data.Temp3pm, rangos, labels=nombres)

'''
columna RainToday
'''
data.RainToday.replace(['No', 'Yes'], [0, 1], inplace=True)
'''
columna RainTomorrow
'''
data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)

data.dropna(axis=0,how='any', inplace=True)

#Partir la tabla en dos
data_train = data[:900]
data_test = data[900:]
x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)

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


# MAQUINA DE SOPORTE VECTORIAL CON VALIDACIÓN CRUZADA
kfold_svc = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_test_train = []
svc = SVC(gamma='auto')
for train, test in kfold_svc.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred_svc= svc.predict(x_test_out)

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
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_svc)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred_svc)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución en MAQUINA DE SOPORTE VECTORIAL ")

precision_svc = precision_score(y_test_out, y_pred_svc, average=None).mean()
print(f'Precisión: {precision_svc}')

recall_svc = recall_score(y_test_out, y_pred_svc, average=None).mean()
print(f'Re-call: {recall_svc}')


# ARBOL DE DECISIÓN con validacion cruzada

kfold_tree = KFold(n_splits=10)
acc_scores_train_train = []
acc_scores_test_train = []
arbol = DecisionTreeClassifier()
# Entreno el modelo
for train, test in kfold_tree.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred_tree= arbol.predict(x_test_out)


# MÉTRICAS

print('*'*50)
print(' ARBOL DECISIÓN Validación cruzada')
# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred_tree)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred_tree)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución con ARBOL DE DECISIÓN")

precision_tree = precision_score(y_test_out, y_pred_tree, average=None).mean()
print(f'Precisión: {precision_tree}')

recall_tree = recall_score(y_test_out, y_pred_tree, average=None).mean()
print(f'Re-call: {recall_tree}')















