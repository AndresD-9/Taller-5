# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar el dataset
data = pd.read_csv('hotel_bookings.csv')
data.head()

# Selección de características relevantes
selected_features = ['lead_time','is_canceled']

# Filtrar el dataset con las características seleccionadas
data_filtered = data[selected_features]

# Dividir en variables independientes (X) y dependientes (y)
X = data_filtered.drop('is_canceled', axis=1)
y = data_filtered['is_canceled']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#SVM Líneal
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
linear_predictions = svm_linear.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_predictions)
print(f'Precisión SVM Kernel Líneal: {linear_accuracy:.2f}')

#SVM Polinómico
svm_poly = SVC(kernel='poly', degree=3) #Se puede cambiar el grado
svm_poly.fit(X_train, y_train)
poly_predictions = svm_poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_predictions)
print(f'Precisión SVM Kernel Polinómico: {poly_accuracy:.2f}')

#SVM RBF
svm_rbf = SVC(kernel= 'rbf')
svm_rbf.fit(X_train, y_train)
rbf_predictions = svm_rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)
print(f'Precisión SVM Kernel RBF: {rbf_accuracy:.2f}')

#Redes bayesianas
bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)

bayes_predictions = bayes_model.predict(X_test)
bayes_accuracy = accuracy_score(y_test, bayes_predictions)

print(f'Precisión de la Red Bayesiana: {bayes_accuracy:.2f}')