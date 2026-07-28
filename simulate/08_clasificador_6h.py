"""
Fase 2, paso 8 - Clasificador binario 6h.

Entrada:
    data/processed/dataset_6h.csv

Salida:
    simulate/models/clf_6h.joblib

Uso:
    python 08_clasificador_6h.py
"""

from pathlib import Path
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
SIMULATE_MODELS_DIR = BASE_DIR / "simulate" / "models"

DATASET_PATH = DATA_PROCESSED_DIR / "dataset_6h.csv"
MODEL_PATH = SIMULATE_MODELS_DIR / "clf_6h.joblib"

TARGET_COL = "target_6h"
SPLIT_COL = "split"
TIMESTAMP_COL = "timestamp"


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


def imprimir_baseline_siempre_negativo(y_true):
    y_pred = [0] * len(y_true)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    print("\n=== PASO 2 - Baseline ingenuo (siempre negativo) ===")
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision clase positiva: {prec:.6f}")
    print(f"Recall clase positiva: {rec:.6f}")
    print(
        "AVISO: accuracy es engañosa en desbalance; el criterio real es recall/precision de la clase positiva."
    )
    if rec == 0:
        print("AVISO EXPLICITO: el baseline no detecta ningun evento fuerte (recall = 0), por lo que es inutil para alerta.")
    return y_pred


def imprimir_matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tabla = pd.DataFrame(
        cm,
        index=["real_0", "real_1"],
        columns=["pred_0", "pred_1"],
    )
    print("\n=== Matriz de confusion ===")
    print(tabla.to_string())


def imprimir_reporte_clasificacion(y_true, y_pred):
    print("\n=== classification_report ===")
    print(classification_report(y_true, y_pred, digits=6, zero_division=0))


def analizar_importancias(modelo, feature_cols):
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": modelo.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")

    print("\n=== Feature importance (descendente) ===")
    for _, row in imp.iterrows():
        print(f"{row['feature']}: {row['importance']:.6f}")

    return imp


def barrer_umbrales(proba_test, y_true, umbrales):
    filas = []
    for umbral in umbrales:
        y_pred = (proba_test > umbral).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = 0.0
        if rec + prec > 0:
            f1 = 2 * prec * rec / (prec + rec)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        filas.append(
            {
                "umbral": umbral,
                "recall": rec,
                "precision": prec,
                "f1": f1,
                "fn": fn,
                "fp": fp,
                "tp": tp,
            }
        )

    tabla = pd.DataFrame(filas)
    print("\n=== Barrido de umbrales sobre probabilidades ===")
    print(tabla[["umbral", "recall", "precision", "fn", "fp", "tp"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    candidatas = tabla[tabla["recall"] >= 0.85].copy()
    if not candidatas.empty:
        recomendada = candidatas.sort_values(["umbral", "recall"], ascending=[True, False]).iloc[0]
        print(
            "\nUMBRAL RECOMENDADO PARA DESPLIEGUE (prioriza deteccion sobre falsas alarmas): "
            f"umbral={recomendada['umbral']:.2f} | recall={recomendada['recall']:.6f} | "
            f"precision={recomendada['precision']:.6f} | f1={recomendada['f1']:.6f} | "
            f"FN={int(recomendada['fn'])} | FP={int(recomendada['fp'])} | TP={int(recomendada['tp'])}"
        )
    else:
        mejor_recall = tabla.sort_values(["recall", "umbral"], ascending=[False, True]).iloc[0]
        print(
            "\nLIMITACION A DOCUMENTAR: ningun umbral del barrido alcanza recall >= 0.85. "
            f"El maximo recall alcanzable en [0.1, 0.5] es {mejor_recall['recall']:.6f} con umbral={mejor_recall['umbral']:.2f}, "
            f"precision={mejor_recall['precision']:.6f}, f1={mejor_recall['f1']:.6f}, "
            f"FN={int(mejor_recall['fn'])}, FP={int(mejor_recall['fp'])}, TP={int(mejor_recall['tp'])}."
        )

    return tabla


def main():
    print("=== PASO 1 - Preparar target binario ===")
    df = cargar_datos()
    feature_cols, p90, train, test, x_train, y_train, x_test, y_test = preparar_datos(df)

    positivos_train = int(y_train.sum())
    positivos_test = int(y_test.sum())

    print(f"Archivo dataset: {DATASET_PATH}")
    print(f"n train: {len(train)} | n test: {len(test)}")
    print(f"n features: {len(feature_cols)}")
    print(f"p90 calculado solo en train: {p90:.6f}")
    print(f"Positivos en train (label_6h=1): {positivos_train}")
    print(f"Positivos en test (label_6h=1): {positivos_test}")
    print(f"features: {feature_cols}")

    y_pred_baseline = imprimir_baseline_siempre_negativo(y_test)

    print("\n=== PASO 3 - Entrenar RandomForestClassifier ===")
    modelo = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
    )
    modelo.fit(x_train, y_train)

    print("\n=== PASO 4 - Evaluar en test ===")
    y_pred = modelo.predict(x_test)

    reporte = classification_report(y_test, y_pred, digits=6, zero_division=0)
    print(reporte)
    imprimir_matriz_confusion(y_test, y_pred)

    rec_modelo = recall_score(y_test, y_pred, zero_division=0)
    rec_baseline = recall_score(y_test, y_pred_baseline, zero_division=0)
    prec_modelo = precision_score(y_test, y_pred, zero_division=0)
    prec_baseline = precision_score(y_test, y_pred_baseline, zero_division=0)
    acc_modelo = accuracy_score(y_test, y_pred)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)

    print("\n=== Comparacion contra baseline siempre negativo ===")
    print(f"Baseline -> accuracy={acc_baseline:.6f}, precision={prec_baseline:.6f}, recall={rec_baseline:.6f}")
    print(f"RF       -> accuracy={acc_modelo:.6f}, precision={prec_modelo:.6f}, recall={rec_modelo:.6f}")
    print(f"Delta recall RF - baseline: {rec_modelo - rec_baseline:+.6f}")

    if rec_modelo <= rec_baseline:
        print("AVISO EXPLICITO: el clasificador no supera al baseline en recall de clase positiva.")

    proba_test = modelo.predict_proba(x_test)[:, 1]
    barrer_umbrales(proba_test, y_test, [0.5, 0.4, 0.3, 0.2, 0.15, 0.1])

    print("\n=== PASO 6 - Feature importance ===")
    analizar_importancias(modelo, feature_cols)

    print("\n=== PASO 5 - Guardar ===")
    os.makedirs(SIMULATE_MODELS_DIR, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")

    print("\n=== PASO 6 - Verificacion fisica en disco ===")
    existe = MODEL_PATH.exists()
    print(f"Existe archivo: {existe}")
    if not existe:
        raise FileNotFoundError(f"No se encontro el modelo en disco: {MODEL_PATH}")

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"Tamano archivo: {size_kb:.2f} KB")


if __name__ == "__main__":
    main()
