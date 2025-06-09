import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("datos_battle_royale.csv")

resumen = df.groupby("perfil")["rendimiento"].describe()
print("Resumen estadístico por perfil:")
print(resumen)


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="perfil", y="rendimiento", palette="Set2")
plt.title("Figura 2. Comparación del rendimiento por perfil de agente")
plt.ylabel("Rendimiento (ticks sobrevividos)")
plt.xlabel("Perfil")
plt.savefig("comparacion_perfiles.png")
plt.close()
