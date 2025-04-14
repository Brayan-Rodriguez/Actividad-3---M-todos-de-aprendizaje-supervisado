import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Leer el dataset
df = pd.read_csv("dataset.csv")

# Codificar texto a números
df['Inicio'] = df['Inicio'].map({'A': 0, 'B': 1, 'C': 2})
df['Destino'] = df['Destino'].map({'A': 0, 'B': 1, 'C': 2})
df['Congestion'] = df['Congestion'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2})

# Separar variables
X = df[['Inicio', 'Destino', 'Hora', 'Congestion']]
y = df['Tiempo (minutos)']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir y evaluar
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Error cuadrático medio (MSE): {mse:.2f}")
for i in range(len(y_pred)):
    print(f"Predicción: {y_pred[i]:.2f}, Real: {y_test.iloc[i]}")
