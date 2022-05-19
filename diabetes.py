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
# Leer csv diabetes
path = './dataSets/diabetes/dataSet.csv'
data = pd.read_csv(path)
#Limpiar y normalizar la data
'''
columna Glucose
'''
data.Glucose.replace(np.nan, 120, inplace=True)
rangos = [ 70, 100 ,120, 150, 170, 200]
nombres = ['1', '2', '3', '4', '5']
data.Glucose = pd.cut(data.Glucose, rangos, labels=nombres)
'''
columna Age
'''
rangos = [ 20, 30, 40, 50, 70, 100]
nombres = ['1', '2', '3', '4', '5']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
'''
columna BMI
'''
data.BMI.replace(np.nan, 32, inplace=True)
rangos = [ 10, 20, 30, 40, 50, 70]
nombres = ['1', '2', '3', '4', '5']
data.BMI = pd.cut(data.BMI, rangos, labels=nombres)
'''
columna DiabetesPedigreeFunction
'''
rangos = [ 0.05, 0.25, 0.50, 1, 1.50, 2.50]
nombres = ['1', '2', '3', '4', '5']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rangos, labels=nombres)
'''
columna BloodPressure
'''
rangos = [ 0, 20, 40, 60, 80, 100, 130]
nombres = ['1', '2', '3', '4', '5', '6']
data.BloodPressure = pd.cut(data.BloodPressure, rangos, labels=nombres)
'''
columna SkinThickness
'''
rangos = [ 0, 20, 40, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']
data.SkinThickness = pd.cut(data.SkinThickness, rangos, labels=nombres)
'''
columna Insulin
'''
rangos = [ 0, 100, 200, 300, 400, 500, 700, 900]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Insulin = pd.cut(data.Insulin, rangos, labels=nombres)

#Borrar NaN
data.dropna(axis=0,how='any', inplace=True)
#Dropear los datos
data.drop(['Pregnancies'], axis= 1, inplace = True)
#Partir la tabla en dos
data_train = data[:383]
data_test = data[383:]
x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)

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
# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


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
