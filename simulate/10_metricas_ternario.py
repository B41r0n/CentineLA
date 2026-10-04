"""
Fase 7 - Métricas para umbrales ternarios (0.30 / 0.70).

Reusa la misma definición de etiqueta y split cronológico (corte fijo en la columna split del dataset) de 08_clasificador_6h.py,
carga simulate/models/clf_6h.joblib e imprime recall y precisión sobre el split de test
para umbrales 0.20, 0.30 y 0.70.

Control: el resultado en 0.20 debe reproducir ≈0.753 / ≈0.415 (corrida de retrain del 01-oct).
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
SIMULATE_MODELS_DIR = BASE_DIR / "simulate" / "models"

DATASET_PATH = DATA_PROCESSED_DIR / "dataset_6h.csv"
MODEL_PATH = SIMULATE_MODELS_DIR / "clf_6h.joblib"

TARGET_COL = "target_6h"
SPLIT_COL = "split"
TIMESTAMP_COL = "timestamp"

UMBRALES = [0.20, 0.30, 0.70]


def cargar_datos():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def preparar_datos(df):
    feature_cols = [
        col
        for col in df.columns
        if col not in {TIMESTAMP_COL, TARGET_COL, SPLIT_COL}
    ]

    train = df[df[SPLIT_COL] == "train"].copy()
    test = df[df[SPLIT_COL] == "test"].copy()

    if train.empty or test.empty:
        raise ValueError("Split invalido: train o test quedo vacio")

    p90 = float(train[TARGET_COL].quantile(0.90))
    train["label_6h"] = (train[TARGET_COL] > p90).astype(int)
    test["label_6h"] = (test[TARGET_COL] > p90).astype(int)

    x_train = train[feature_cols]
    y_train = train["label_6h"]
    x_test = test[feature_cols]
    y_test = test["label_6h"]

    return feature_cols, p90, train, test, x_train, y_train, x_test, y_test


def main():
    print("=== Métricas para umbrales ternarios ===")
    df = cargar_datos()
    feature_cols, p90, train, test, x_train, y_train, x_test, y_test = preparar_datos(df)

    print(f"Dataset: {DATASET_PATH}")
    print(f"n train: {len(train)} | n test: {len(test)}")
    print(f"n features: {len(feature_cols)}")
    print(f"p90 (solo train): {p90:.6f}")
    print(f"Positivos en test (label_6h=1): {int(y_test.sum())}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

    modelo = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")

    proba_test = modelo.predict_proba(x_test)[:, 1]

    print("\n=== Métricas por umbral (split test cronológico) ===")
    print(f"{'Umbral':>7}  {'Recall':>10}  {'Precision':>10}")
    print("-" * 32)

    resultados = {}
    for umbral in UMBRALES:
        y_pred = (proba_test > umbral).astype(int)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        resultados[umbral] = (rec, prec)
        print(f"{umbral:>7.2f}  {rec:>10.6f}  {prec:>10.6f}")

    # Control: verificar que 0.20 reproduce los valores esperados
    rec_20, prec_20 = resultados[0.20]
    print("\n=== Control (umbral 0.20) ===")
    print(f"Recall obtenido:    {rec_20:.6f}  (esperado ~0.752998)")
    print(f"Precision obtenida: {prec_20:.6f}  (esperado ~0.415344)")

    diff_rec = abs(rec_20 - 0.752998)
    diff_prec = abs(prec_20 - 0.415344)
    if diff_rec > 0.01 or diff_prec > 0.01:
        print(f"\nADVERTENCIA: Diferencia > 0.01 respecto a la corrida de referencia.")
        print(f"   Delta recall = {diff_rec:.6f}, Delta precision = {diff_prec:.6f}")
        print("   NO editar app.py con estos valores. Revisar consistencia del dataset/modelo.")
        return 1
    else:
        print("\nControl OK: valores reproducen la referencia (diferencia <= 0.01).")
        print("  Seguro para actualizar app.py con las metricas ternarias.")
        return 0


if __name__ == "__main__":
    exit(main())