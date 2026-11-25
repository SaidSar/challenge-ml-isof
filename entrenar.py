# ==============================================================================
# ENTRENAMIENTO: SISTEMA PREDICTIVO DE MANTENIMIENTO INDUSTRIAL
# Challenge Lab - ISOF CLASS
# Lee sensores_planta.csv, entrena un RandomForestClassifier y exporta
# modelo_mantenimiento.pkl
# ==============================================================================

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Las rutas se calculan respecto a este archivo, asi funciona desde cualquier carpeta
CARPETA = Path(__file__).resolve().parent
ARCHIVO_CSV = CARPETA / "sensores_planta.csv"
ARCHIVO_MODELO = CARPETA / "modelo_mantenimiento.pkl"

COLUMNAS_ENTRADA = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
COLUMNA_OBJETIVO = "estado"

# 1. Cargar el CSV con pandas
print(f"1. Cargando datos desde '{ARCHIVO_CSV.name}'...")
try:
    df = pd.read_csv(ARCHIVO_CSV, sep=",")
except FileNotFoundError:
    sys.exit(f"Error: no se encontro el archivo '{ARCHIVO_CSV}'.")

df.columns = df.columns.str.strip().str.lower()
print(f"   Registros leidos: {len(df)}")

# 2. Limpieza: eliminar filas con valores nulos
df = df.dropna()
print(f"   Registros despues de dropna(): {len(df)}")

faltantes = [c for c in COLUMNAS_ENTRADA + [COLUMNA_OBJETIVO] if c not in df.columns]
if faltantes:
    sys.exit(f"Error: faltan columnas en el CSV: {faltantes}")

if len(df) < 30:
    print("Aviso: el reto pide al menos 30 registros.")

# 3. Variables de entrada (X) y objetivo (y)
X = df[COLUMNAS_ENTRADA]
y = df[COLUMNA_OBJETIVO].astype(int)

# 4. Division 80% entrenamiento / 20% prueba (estratificada para conservar las 3 clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar el RandomForestClassifier
print("2. Entrenando RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 6. Evaluar la precision
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"Precision del modelo (Accuracy): {precision * 100:.2f}%")

if precision < 0.80:
    sys.exit("La precision es menor a 80%. Revisa o amplia el dataset antes de exportar.")

# 7. Exportar el modelo con joblib
joblib.dump(modelo, ARCHIVO_MODELO)
print(f"3. Modelo exportado como '{ARCHIVO_MODELO.name}'.")
