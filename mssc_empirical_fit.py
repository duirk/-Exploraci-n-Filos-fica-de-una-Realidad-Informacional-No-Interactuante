import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# 1. Definición de modelos para comparación (Protocolo de Falsabilidad)
def modelo_mssc(L, alpha, beta):
    return alpha * np.tanh(beta * L)

def modelo_lineal(L, m, c):
    return m * L + c

# 2. Generación de datos sintéticos con ruido controlado
np.random.seed(42) # Para reproducibilidad (esencial en ciencia)
L_val = np.linspace(0.1, 10, 100)
V_real = 2.0 * np.tanh(0.5 * L_val) + np.random.normal(0, 0.05, 100)

# 3. Optimización y Ajuste
params_mssc, _ = curve_fit(modelo_mssc, L_val, V_real)
params_lin, _ = curve_fit(modelo_lineal, L_val, V_real)

# 4. Análisis de métricas de calidad
V_pred_mssc = modelo_mssc(L_val, *params_mssc)
r2_mssc = r2_score(V_real, V_pred_mssc)
rmse_mssc = np.sqrt(mean_squared_error(V_real, V_pred_mssc))

print(f"--- Análisis Estadístico MSSC ---")
print(f"Alpha: {params_mssc[0]:.4f}, Beta: {params_mssc[1]:.4f}")
print(f"R²: {r2_mssc:.4f} | RMSE: {rmse_mssc:.4f}")

# 5. Visualización técnica
plt.figure(figsize=(10, 6))
plt.scatter(L_val, V_real, label='Datos Observados (Simulados)', alpha=0.5, color='gray', s=15)
plt.plot(L_val, V_pred_mssc, label=f'Ajuste MSSC (R²={r2_mssc:.3f})', color='red', linewidth=2)
plt.plot(L_val, modelo_lineal(L_val, *params_lin), '--', label='Modelo Lineal (Control)', color='blue', alpha=0.6)

plt.title('Validación de Saturación: MSSC vs Modelo Lineal')
plt.xlabel('Intensidad de Información (L)')
plt.ylabel('Nivel de Activación (V)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('validacion_mssc.png', dpi=300) # Guardar figura para el repo
plt.show()