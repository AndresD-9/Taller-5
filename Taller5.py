# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC

# Cargar el dataset
data = pd.read_csv('hotel_bookings.csv')

# Selección de características relevantes
selected_features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 
                     'adults', 'children', 'babies', 'adr', 'is_canceled']

# Filtrar el dataset con las características seleccionadas
data_filtered = data[selected_features]

# Reemplazar valores faltantes en la columna 'children' usando .loc para evitar advertencias
data_filtered.loc[:, 'children'] = data_filtered['children'].fillna(0)

# Dividir en variables independientes (X) y dependientes (y)
X = data_filtered.drop('is_canceled', axis=1)
y = data_filtered['is_canceled']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos para SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Aplicar Máquina de Vectores de Soporte (SVM) ###
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Predecir con el conjunto de prueba
y_pred_svm = svm.predict(X_test_scaled)

# Evaluar la precisión del SVM
print("Precisión de SVM:", metrics.accuracy_score(y_test, y_pred_svm))

# Mostrar la matriz de confusión para SVM

ConfusionMatrixDisplay.from_estimator(svm, X_test_scaled, y_test)
plt.title('Matriz de confusión - SVM')
plt.show()

scores = cross_val_score(svm, X, y, cv=40)
print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
"""
### Aplicar Redes Bayesianas ###
# Definir la estructura del modelo usando BayesianNetwork
model = BayesianNetwork([('lead_time', 'is_canceled'), ('stays_in_weekend_nights', 'is_canceled'), 
                         ('stays_in_week_nights', 'is_canceled'), ('adr', 'is_canceled')])

# Estimar los parámetros utilizando el estimador bayesiano
model.fit(data_filtered, estimator=BayesianEstimator, prior_type="BDeu")

# Inferencia
inference = VariableElimination(model)

# Consulta ejemplo (probabilidad de cancelación dadas algunas evidencias)
q = inference.query(variables=['is_canceled'], evidence={'lead_time': 100, 'adr': 150})
print(q)
"""