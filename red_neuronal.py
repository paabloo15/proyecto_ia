# red_neuronal.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar datos
df = pd.read_csv("datos_battle_royale.csv")

# Codificar perfil
df['perfil'] = LabelEncoder().fit_transform(df['perfil'])

# Separar características y etiqueta
X = df.drop('rendimiento', axis=1)
y = df['rendimiento']  # regresión

# Escalado
scaler = StandardScaler()
X = scaler.fit_transform(X)

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Red neuronal
model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1)  # regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluación
loss, mae = model.evaluate(X_test, y_test)
print(f"Pérdida (MSE): {loss:.2f}, MAE: {mae:.2f}")
