import pandas as pd
from skrebate import ReliefF
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# cargar el conjunto de datos de Iris
data = load_iris()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.Series(data['target'])

# separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crear un objeto ReliefF y seleccionar las características
fs = ReliefF(n_features_to_select=2, n_neighbors=10)
X_train_fs = fs.fit_transform(X_train.values, y_train)

# obtener los nombres de las características seleccionadas
selected_features = X.columns[fs.top_features_]

# imprimir las características seleccionadas
print(selected_features)