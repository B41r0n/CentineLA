# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Fase 2, paso 9 - Gateway simulado en modo streaming.

Objetivo:
    Simular un gateway RPi4 procesando la serie fila por fila (orden cronologico),
    calculando features sin fuga de datos y aplicando el clasificador 6h.

Entradas:
    data/processed/proxy_q_la_honda.csv
    data/processed/dataset_6h.csv
    simulate/models/clf_6h.joblib

Salida:
    data/processed/log_gateway_simulado.csv

Uso:
    python 09_gateway_simulado.py
"""

from collections import deque
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "simulate" / "models"

PROXY_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
DATASET_PATH = DATA_PROCESSED_DIR / "dataset_6h.csv"
MODEL_PATH = MODELS_DIR / "clf_6h.joblib"
LOG_PATH = DATA_PROCESSED_DIR / "log_gateway_simulado.csv"

LOW_QUALITY_MIN_ESTACIONES = 2
EPSILON = 1e-6
N_VALIDACION_TS = 500
UMBRAL_PRECAUCION = 0.30
UMBRAL_ALERTA = 0.70


def clasificar_estado_ternario(proba: float) -> str:
    """Sistema ternario de alertas: NORMAL / PRECAUCIÓN / ALERTA."""
    if proba >= UMBRAL_ALERTA:
        return "ALERTA"
    elif proba >= UMBRAL_PRECAUCION:
        return "PRECAUCIÓN"
    return "NORMAL"


def cargar_entradas():
    if not PROXY_PATH.exists():
        raise FileNotFoundError(f"No existe: {PROXY_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe: {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe: {MODEL_PATH}")

    proxy = pd.read_csv(PROXY_PATH, parse_dates=["timestamp"])
    proxy = proxy.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    dataset = dataset.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    modelo = joblib.load(MODEL_PATH)
    if hasattr(modelo, "n_jobs"):
        modelo.set_params(n_jobs=1)
    return proxy, dataset, modelo


def construir_base_stream(proxy_completo):
    # Replica la logica del ETL batch: filtrar calidad minima y reindexar horario completo.
    proxy = proxy_completo.copy()
    proxy = proxy[proxy["n_estaciones_disponibles"] >= LOW_QUALITY_MIN_ESTACIONES].copy()

    base = proxy[["timestamp", "P_basin", "Q_scs_proxy"]].copy()
    base = base.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    base = base.set_index("timestamp")

    full_index = pd.date_range(base.index.min(), base.index.max(), freq="1h")
    base = base.reindex(full_index)
    base.index.name = "timestamp"
    return base.reset_index()


def _lag(serie_deque, horas):
    if len(serie_deque) <= horas:
        return np.nan
    return serie_deque[-(horas + 1)]


def _rolling_sum(serie_deque, horas):
    if not serie_deque:
        return np.nan
    ventana = np.array(list(serie_deque)[-horas:], dtype=float)
    valid = ~np.isnan(ventana)
    if valid.sum() == 0:
        return np.nan
    return float(np.nansum(ventana))


def _rolling_max(serie_deque, horas):
    if not serie_deque:
        return np.nan
    ventana = np.array(list(serie_deque)[-horas:], dtype=float)
    valid = ~np.isnan(ventana)
    if valid.sum() == 0:
        return np.nan
    return float(np.nanmax(ventana))


def calcular_features_streaming(base_stream):
    p_hist = deque(maxlen=72)
    q_hist = deque(maxlen=72)
    registros = []

    for row in base_stream.itertuples(index=False):
        ts = row.timestamp
        p_now = float(row.P_basin) if pd.notna(row.P_basin) else np.nan
        q_now = float(row.Q_scs_proxy) if pd.notna(row.Q_scs_proxy) else np.nan

        p_hist.append(p_now)
        q_hist.append(q_now)

        feat = {
            "timestamp": ts,
            "P_basin": p_now,
            "Q_actual": q_now,
            "lag_1h": _lag(p_hist, 1),
            "lag_3h": _lag(p_hist, 3),
            "lag_6h": _lag(p_hist, 6),
            "lag_12h": _lag(p_hist, 12),
            "lag_24h": _lag(p_hist, 24),
            "Q_lag_1h": _lag(q_hist, 1),
            "Q_lag_3h": _lag(q_hist, 3),
            "Q_lag_6h": _lag(q_hist, 6),
            "roll_sum_3h": _rolling_sum(p_hist, 3),
            "roll_sum_6h": _rolling_sum(p_hist, 6),
            "roll_sum_12h": _rolling_sum(p_hist, 12),
            "roll_sum_24h": _rolling_sum(p_hist, 24),
            "roll_sum_48h": _rolling_sum(p_hist, 48),
            "roll_max_6h": _rolling_max(p_hist, 6),
            "hora_dia": int(ts.hour),
            "mes": int(ts.month),
        }
        registros.append(feat)

    out = pd.DataFrame(registros)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def validar_alineacion_features(stream_features, dataset_batch, feature_cols):
    if dataset_batch.empty:
        raise ValueError("dataset_6h.csv esta vacio; no se puede validar.")

    n = min(N_VALIDACION_TS, len(dataset_batch))
    muestra = dataset_batch.sample(n=n, random_state=42).copy()
    merged = muestra[["timestamp"] + feature_cols].merge(
        stream_features[["timestamp"] + feature_cols],
        on="timestamp",
        how="left",
        suffixes=("_batch", "_stream"),
    )

    faltantes_stream = merged[[f"{c}_stream" for c in feature_cols]].isna().all(axis=1).sum()
    if faltantes_stream > 0:
        print(
            f"ERROR VALIDACION: {faltantes_stream} timestamps de la muestra no existen en features streaming."
        )
        return False

    errores = []
    for row in merged.itertuples(index=False):
        ts = row.timestamp
        for col in feature_cols:
            v_batch = getattr(row, f"{col}_batch")
            v_stream = getattr(row, f"{col}_stream")

            if pd.isna(v_batch) and pd.isna(v_stream):
                continue
            if pd.isna(v_batch) != pd.isna(v_stream):
                errores.append((ts, col, v_batch, v_stream, "NaN_mismatch"))
                continue

            diff = abs(float(v_batch) - float(v_stream))
            if diff > EPSILON:
                errores.append((ts, col, v_batch, v_stream, diff))

    if errores:
        print(
            f"ERROR VALIDACION: {len(errores)} diferencias > {EPSILON} entre batch y streaming."
        )
        print("Primeros 20 errores:")
        for e in errores[:20]:
            print(
                f"timestamp={e[0]}, feature={e[1]}, batch={e[2]}, stream={e[3]}, detalle={e[4]}"
            )
        return False

    print(
        f"VALIDACION OK: {n} timestamps aleatorios comparados, sin diferencias > {EPSILON}."
    )
    return True


def _procesar_lote(modelo, feature_cols, lote_x, lote_meta, logs):
    if not lote_x:
        return

    x_df = pd.DataFrame(lote_x, columns=feature_cols)
    probas = modelo.predict_proba(x_df)[:, 1]

    for meta, proba in zip(lote_meta, probas):
        estado = clasificar_estado_ternario(float(proba))
        registro = {
            "timestamp": meta["timestamp"],
            "P_basin": meta["P_basin"],
            "Q_actual": meta["Q_actual"],
            "Q_lag_1h": meta["Q_lag_1h"],
            "roll_sum_24h": meta["roll_sum_24h"],
            "hora_dia": meta["hora_dia"],
            "mes": meta["mes"],
            "proba_alerta": float(proba),
            "estado": estado,
        }
        logs.append(registro)


def simular_gateway(stream_features, modelo, feature_cols):
    logs = []
    lote_x = []
    lote_meta = []
    batch_size = 2048

    for row in stream_features.itertuples(index=False):
        x = np.array([getattr(row, c) for c in feature_cols], dtype=float)
        if np.isnan(x).any():
            continue

        lote_x.append(x)
        lote_meta.append(
            {
                "timestamp": row.timestamp,
                "P_basin": float(row.P_basin),
                "Q_actual": float(row.Q_actual),
                "Q_lag_1h": float(row.Q_lag_1h),
                "roll_sum_24h": float(row.roll_sum_24h),
                "hora_dia": int(row.hora_dia),
                "mes": int(row.mes),
            }
        )

        if len(lote_x) >= batch_size:
            _procesar_lote(modelo, feature_cols, lote_x, lote_meta, logs)
            lote_x = []
            lote_meta = []

    _procesar_lote(modelo, feature_cols, lote_x, lote_meta, logs)

    return pd.DataFrame(logs)


def main():
    print("=== PASO 1 - Cargar entradas ===")
    proxy, dataset, modelo = cargar_entradas()
    feature_cols = [
        c
        for c in dataset.columns
        if c not in {"timestamp", "target_6h", "split"}
    ]

    print(f"Proxy cargado: {PROXY_PATH} | filas={len(proxy)}")
    print(f"Dataset batch cargado: {DATASET_PATH} | filas={len(dataset)}")
    print(f"Modelo cargado: {MODEL_PATH}")
    print(f"n features esperadas: {len(feature_cols)}")

    print("\n=== PASO 2 - Motor de features en streaming ===")
    base_stream = construir_base_stream(proxy)
    stream_features = calcular_features_streaming(base_stream)
    print(f"Filas en stream horario tras filtro/reindex: {len(stream_features)}")

    ok = validar_alineacion_features(stream_features, dataset, feature_cols)
    if not ok:
        print(
            "ABORTADO: bug de alineacion temporal detectado entre batch y streaming. "
            "No se ejecuta inferencia hasta resolverlo."
        )
        return

    print("\n=== PASO 3 - Inferencia streaming con umbral 0.20 ===")
    log = simular_gateway(stream_features, modelo, feature_cols)
    if log.empty:
        raise ValueError("El log simulado quedo vacio; revisar disponibilidad de features completas.")

    print(f"Filas inferidas (sin NaN en features): {len(log)}")

    print("\n=== PASO 4 - Guardar salida y resumen ===")
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    log.to_csv(LOG_PATH, index=False)
    print(f"Log guardado: {LOG_PATH}")

    total_alertas = int((log["estado"] == "ALERTA").sum())
    total_normal = int((log["estado"] == "NORMAL").sum())
    print(f"Total ALERTA: {total_alertas}")
    print(f"Total NORMAL: {total_normal}")

    dist_anual = (
        log.assign(anio=log["timestamp"].dt.year)
        .groupby(["anio", "estado"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print("\nDistribucion temporal por anio:")
    print(dist_anual.to_string())

    # Referencia operativa esperada (orden de magnitud): alrededor de 1k alertas en test.
    referencia = 299 + 865
    ratio = total_alertas / referencia if referencia > 0 else np.nan
    print(
        f"\nChequeo orden de magnitud vs referencia (~{referencia}): "
        f"alertas={total_alertas}, ratio={ratio:.3f}"
    )

    print("\n=== PASO 5 - Verificacion de archivo guardado ===")
    check = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    print(f"Lectura real de log: filas={len(check)}")
    print(f"ALERTA en archivo: {(check['estado'] == 'ALERTA').sum()}")


if __name__ == "__main__":
    main()
