import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


df = pd.read_csv("datos_battle_royale.csv")


X = df[['salud', 'recursos', 'enemigos_cerca']]
y = df['rendimiento']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nPérdida (MSE): {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f} (más cerca de 1 = mejor ajuste)")


print("\nPrimeros 10 valores reales vs. predichos:")
for i in range(10):
    print(f"Real: {y_test.values[i]:.2f} - Predicho: {y_pred[i]:.2f}")

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title("Evolución de la pérdida (MSE)")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Rendimiento real")
plt.ylabel("Rendimiento predicho")
plt.title("Rendimiento real vs predicho")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.grid(True)
plt.show()
