import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch, BicScore

# Carga el archivo CSV
data = pd.read_csv('hotel_bookings.csv')

# Define la estructura de la red
# Por ejemplo, A -> B -> C indica que A es padre de B, y B es padre de C
model = BayesianModel([('A', 'B'), ('B', 'C')])

# Ajusta el modelo a los datos
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Realiza inferencias
infer = VariableElimination(model)
query_result = infer.query(variables=['C'], evidence={'A': 1})
print(query_result)

# Usa HillClimbSearch para encontrar la mejor estructura
est = HillClimbSearch(data)
best_model = est.estimate(scoring_method=BicScore(data))

print(best_model.edges())