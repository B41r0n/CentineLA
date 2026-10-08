# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Fase 2, paso 7 - Entrenamiento y evaluacion del modelo 6h.

Entrada principal:
    data/processed/dataset_6h.csv

Entrada para baseline de persistencia:
    data/processed/proxy_q_la_honda.csv

Salida:
    simulate/models/rf_6h.joblib

Uso:
    python 07_entrenar_modelo_6h.py
"""

from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
SIMULATE_MODELS_DIR = BASE_DIR / "simulate" / "models"

DATASET_PATH = DATA_PROCESSED_DIR / "dataset_6h.csv"
PROXY_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
MODEL_PATH = SIMULATE_MODELS_DIR / "rf_6h.joblib"
MODEL_DELTA_PATH = SIMULATE_MODELS_DIR / "rf_6h_delta.joblib"
MODEL_WEIGHTED_PATH = SIMULATE_MODELS_DIR / "rf_6h_weighted.joblib"

TARGET_COL = "target_6h"
SPLIT_COL = "split"
TIMESTAMP_COL = "timestamp"
BASELINE_COL = "Q_scs_proxy"


def cargar_datos():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {DATASET_PATH}")
    if not PROXY_PATH.exists():
        raise FileNotFoundError(
            f"No existe el archivo requerido para baseline de persistencia: {PROXY_PATH}"
        )

    df = pd.read_csv(DATASET_PATH, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    proxy = pd.read_csv(PROXY_PATH, parse_dates=[TIMESTAMP_COL])
    proxy = proxy[[TIMESTAMP_COL, BASELINE_COL]].drop_duplicates(subset=[TIMESTAMP_COL], keep="last")

    return df, proxy


def preparar_xy_split(df):
    feature_cols = [
        col
        for col in df.columns
        if col not in {TIMESTAMP_COL, TARGET_COL, SPLIT_COL}
    ]

    train = df[df[SPLIT_COL] == "train"].copy()
    test = df[df[SPLIT_COL] == "test"].copy()

    if train.empty or test.empty:
        raise ValueError("Split invalido: train o test quedo vacio")

    x_train = train[feature_cols]
    y_train = train[TARGET_COL]
    x_test = test[feature_cols]
    y_test = test[TARGET_COL]

    return feature_cols, train, test, x_train, y_train, x_test, y_test


def metricas_regresion(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = float("nan")
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def imprimir_reporte(nombre, m_baseline, m_rf):
    print(f"\n=== {nombre} ===")
    print(
        "Baseline persistencia -> "
        f"MAE={m_baseline['MAE']:.6f}, RMSE={m_baseline['RMSE']:.6f}, R2={m_baseline['R2']:.6f}"
    )
    print(
        "RandomForest         -> "
        f"MAE={m_rf['MAE']:.6f}, RMSE={m_rf['RMSE']:.6f}, R2={m_rf['R2']:.6f}"
    )

    d_mae = m_baseline["MAE"] - m_rf["MAE"]
    d_rmse = m_baseline["RMSE"] - m_rf["RMSE"]
    d_r2 = m_rf["R2"] - m_baseline["R2"]
    print(
        "Delta RF vs baseline -> "
        f"MAE={d_mae:+.6f} (positivo es mejor), "
        f"RMSE={d_rmse:+.6f} (positivo es mejor), "
        f"R2={d_r2:+.6f} (positivo es mejor)"
    )


def evaluar_baseline_persistencia(test_df, proxy_df, y_test):
    base_eval = test_df[[TIMESTAMP_COL]].merge(proxy_df, on=TIMESTAMP_COL, how="left")
    n_nan = base_eval[BASELINE_COL].isna().sum()
    if n_nan > 0:
        raise ValueError(
            "El baseline de persistencia quedo con NaN tras merge por timestamp: "
            f"{n_nan} filas sin {BASELINE_COL}"
        )

    y_pred_base = base_eval[BASELINE_COL].to_numpy()
    y_true = y_test.to_numpy()
    return y_true, y_pred_base


def evaluar_subsets_eventos(y_true, y_pred_base, y_pred_rf, y_train):
    p75_train = float(pd.Series(y_train).quantile(0.75))

    mask_gt0 = y_true > 0
    mask_p75 = y_true > p75_train

    resultados = {}
    for nombre, mask in {
        "EVENTOS target_6h > 0": mask_gt0,
        f"EVENTOS target_6h > p75_train ({p75_train:.6f})": mask_p75,
    }.items():
        n = int(mask.sum())
        if n == 0:
            resultados[nombre] = {"n": 0, "baseline": None, "rf": None}
            continue

        m_base = metricas_regresion(y_true[mask], y_pred_base[mask])
        m_rf = metricas_regresion(y_true[mask], y_pred_rf[mask])
        resultados[nombre] = {"n": n, "baseline": m_base, "rf": m_rf}

    return resultados


def evaluar_y_reportar(y_true_test, y_pred_base, y_pred_modelo, y_train, nombre_enfoque):
    print(f"\n=== PASO 4 - Evaluar en test ({nombre_enfoque}) ===")
    met_base_global = metricas_regresion(y_true_test, y_pred_base)
    met_modelo_global = metricas_regresion(y_true_test, y_pred_modelo)
    imprimir_reporte("REPORTE GLOBAL", met_base_global, met_modelo_global)

    resultados_eventos = evaluar_subsets_eventos(
        y_true=y_true_test,
        y_pred_base=y_pred_base,
        y_pred_rf=y_pred_modelo,
        y_train=y_train.to_numpy(),
    )

    for nombre, bloque in resultados_eventos.items():
        if bloque["n"] == 0:
            print(f"\n=== {nombre} ===")
            print("Sin filas en este subset; no se calculan metricas.")
            continue

        print(f"\nFilas en subset: {bloque['n']}")
        imprimir_reporte(nombre, bloque["baseline"], bloque["rf"])

    bloque_eventos_gt0 = resultados_eventos.get("EVENTOS target_6h > 0")
    if bloque_eventos_gt0 and bloque_eventos_gt0["n"] > 0:
        m_base = bloque_eventos_gt0["baseline"]
        m_rf = bloque_eventos_gt0["rf"]
        rf_supera = (
            (m_rf["MAE"] < m_base["MAE"])
            and (m_rf["RMSE"] < m_base["RMSE"])
            and (m_rf["R2"] > m_base["R2"])
        )
        if not rf_supera:
            print(
                "\nADVERTENCIA EXPLICITA: RF NO supera al baseline de persistencia en el subset de eventos (target_6h > 0)."
            )


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

    dinamica_lluvia = imp[
        imp["feature"].str.startswith("lag_")
        | imp["feature"].str.startswith("roll_")
        | imp["feature"].str.startswith("Q_lag_")
        | imp["feature"].isin(["P_basin", "Q_actual"])
    ]
    dinamica_lluvia = pd.concat(
        [
            dinamica_lluvia,
            imp[imp["feature"].isin(["P_basin", "Q_actual"])],
        ],
        axis=0,
    ).drop_duplicates(subset=["feature"])
    temporal = imp[imp["feature"].isin(["hora_dia", "mes"])]

    peso_lluvia = float(dinamica_lluvia["importance"].sum())
    peso_temporal = float(temporal["importance"].sum())

    print("\nChequeo fisico de importancia:")
    print(f"Peso total dinamica lluvia (P_basin + Q_actual + lags + Q_lags + rolling): {peso_lluvia:.6f}")
    print(f"Peso total calendario (hora_dia + mes): {peso_temporal:.6f}")

    if peso_temporal > peso_lluvia:
        print(
            "ADVERTENCIA: domina la senal temporal (hora/mes) sobre lluvia; "
            "riesgo de patrones espurios."
        )
    else:
        print("OK: domina la dinamica de lluvia, consistente con la fisica esperada.")


def main():
    print("=== PASO 1 - Preparar ===")
    df, proxy = cargar_datos()
    feature_cols, train, test, x_train, y_train, x_test, y_test = preparar_xy_split(df)

    print(f"Archivo dataset: {DATASET_PATH}")
    print(f"Archivo proxy baseline: {PROXY_PATH}")
    print(f"n train: {len(train)} | n test: {len(test)}")
    print(f"n features: {len(feature_cols)}")
    print(f"features: {feature_cols}")

    print("\n=== PASO 2 - Baseline ingenuo (persistencia) ===")
    y_true_test, y_pred_base = evaluar_baseline_persistencia(test, proxy, y_test)

    print("\n=== PASO 3A - Entrenar RandomForestRegressor (target absoluto) ===")
    modelo = RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    modelo.fit(x_train, y_train)

    y_pred_rf = modelo.predict(x_test)
    evaluar_y_reportar(
        y_true_test=y_true_test,
        y_pred_base=y_pred_base,
        y_pred_modelo=y_pred_rf,
        y_train=y_train,
        nombre_enfoque="RF target absoluto",
    )

    print("\n=== PASO 1 (weighted) - Sample weights sobre target absoluto ===")
    y_train_max = float(y_train.max())
    if y_train_max <= 0:
        raise ValueError(
            "y_train.max() es <= 0; no se puede aplicar weights_train = 1 + (y_train / y_train.max()) * 20"
        )
    weights_train = 1.0 + (y_train / y_train_max) * 20.0
    print(
        "weights_train stats -> "
        f"min={weights_train.min():.6f}, max={weights_train.max():.6f}, mean={weights_train.mean():.6f}"
    )

    print("\n=== PASO 2 (weighted) - Entrenar RF con sample_weight ===")
    modelo_weighted = RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    modelo_weighted.fit(x_train, y_train, sample_weight=weights_train)

    y_pred_weighted = modelo_weighted.predict(x_test)
    evaluar_y_reportar(
        y_true_test=y_true_test,
        y_pred_base=y_pred_base,
        y_pred_modelo=y_pred_weighted,
        y_train=y_train,
        nombre_enfoque="RF target absoluto ponderado",
    )

    print("\n=== PASO 1 (delta) - Nuevo target de entrenamiento ===")
    if "Q_actual" not in x_train.columns or "Q_actual" not in x_test.columns:
        raise ValueError("No existe la feature Q_actual en X; no se puede entrenar enfoque delta.")
    delta_6h_train = y_train.to_numpy() - x_train["Q_actual"].to_numpy()
    delta_6h_test = y_test.to_numpy() - x_test["Q_actual"].to_numpy()
    print(
        "delta_6h_train stats -> "
        f"min={delta_6h_train.min():.6f}, max={delta_6h_train.max():.6f}, mean={delta_6h_train.mean():.6f}"
    )
    print(
        "delta_6h_test stats  -> "
        f"min={delta_6h_test.min():.6f}, max={delta_6h_test.max():.6f}, mean={delta_6h_test.mean():.6f}"
    )

    print("\n=== PASO 2 (delta) - Entrenar RF sobre delta_6h_train ===")
    modelo_delta = RandomForestRegressor(
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    modelo_delta.fit(x_train, delta_6h_train)

    print("\n=== PASO 3 (delta) - Reconstruir prediccion absoluta ===")
    pred_delta_test = modelo_delta.predict(x_test)
    pred_absoluta = x_test["Q_actual"].to_numpy() + pred_delta_test
    print(
        "pred_absoluta stats -> "
        f"min={pred_absoluta.min():.6f}, max={pred_absoluta.max():.6f}, mean={pred_absoluta.mean():.6f}"
    )

    evaluar_y_reportar(
        y_true_test=y_true_test,
        y_pred_base=y_pred_base,
        y_pred_modelo=pred_absoluta,
        y_train=y_train,
        nombre_enfoque="RF delta reconstruido",
    )

    print("\n=== PASO 5 - Feature importance (target absoluto) ===")
    analizar_importancias(modelo, feature_cols)

    print("\n=== PASO 5 - Feature importance (delta) ===")
    analizar_importancias(modelo_delta, feature_cols)

    print("\n=== PASO 5 - Feature importance (weighted) ===")
    analizar_importancias(modelo_weighted, feature_cols)

    print("\n=== PASO 6 - Guardar modelos ===")
    os.makedirs(SIMULATE_MODELS_DIR, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(modelo_delta, MODEL_DELTA_PATH)
    joblib.dump(modelo_weighted, MODEL_WEIGHTED_PATH)
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Modelo delta guardado en: {MODEL_DELTA_PATH}")
    print(f"Modelo weighted guardado en: {MODEL_WEIGHTED_PATH}")

    print("\n=== PASO 7 - Verificacion fisica en disco ===")
    existe_abs = MODEL_PATH.exists()
    existe_delta = MODEL_DELTA_PATH.exists()
    existe_weighted = MODEL_WEIGHTED_PATH.exists()
    print(f"Existe archivo absoluto: {existe_abs}")
    print(f"Existe archivo delta: {existe_delta}")
    print(f"Existe archivo weighted: {existe_weighted}")
    if not existe_abs:
        raise FileNotFoundError(f"No se encontro el modelo en disco: {MODEL_PATH}")
    if not existe_delta:
        raise FileNotFoundError(f"No se encontro el modelo delta en disco: {MODEL_DELTA_PATH}")
    if not existe_weighted:
        raise FileNotFoundError(f"No se encontro el modelo weighted en disco: {MODEL_WEIGHTED_PATH}")

    size_abs_kb = MODEL_PATH.stat().st_size / 1024
    size_delta_kb = MODEL_DELTA_PATH.stat().st_size / 1024
    size_weighted_kb = MODEL_WEIGHTED_PATH.stat().st_size / 1024
    print(f"Tamano archivo absoluto: {size_abs_kb:.2f} KB")
    print(f"Tamano archivo delta: {size_delta_kb:.2f} KB")
    print(f"Tamano archivo weighted: {size_weighted_kb:.2f} KB")


if __name__ == "__main__":
    main()
