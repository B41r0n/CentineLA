"""
Fase 2, paso 3 - Proxy SCS-CN para La Honda.

Lee los historico_*.csv de data/raw, inspecciona la serie cruda para decidir
la agregacion, limpia por estacion, arma P_basin y calcula el proxy Q_scs.

Uso:
    python 04_scs_cn_proxy.py

Salida:
    ../data/processed/proxy_q_la_honda.csv
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

from cuenca_la_honda_params import CN


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

HISTORICO_FILES = [
    DATA_RAW_DIR / "historico_0027015290.csv",
    DATA_RAW_DIR / "historico_0027015310.csv",
    DATA_RAW_DIR / "historico_0027015330.csv",
]

STATION_META = {
    "historico_0027015290.csv": {
        "codigo": "0027015290",
        "nombre": "Pajarito",
        "sensor_principal": 240,
    },
    "historico_0027015310.csv": {
        "codigo": "0027015310",
        "nombre": "Metromedellin",
        "sensor_principal": 240,
    },
    "historico_0027015330.csv": {
        "codigo": "0027015330",
        "nombre": "Olaya Herrera",
        "sensor_principal": 240,
        "sensor_qa": 257,
    },
}

SENSOR_CANDIDATES = ["codigosensor", "sensor", "id_sensor", "idsensor"]
PRECIP_CANDIDATES = ["valorobservado", "precipitacion", "precip", "lluvia"]
TIME_CANDIDATES = ["fechaobservacion", "fecha_observacion", "timestamp", "datetime", "time"]


def detectar_columna(df, candidatos):
    for nombre in df.columns:
        nombre_norm = str(nombre).strip().lower()
        if nombre_norm in candidatos:
            return nombre
    return None


def inspeccionar_csv(path):
    df = pd.read_csv(path)
    print(f"\n=== {path.name} ===")
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes)
    print("sample 10:")
    print(df.head(10).to_string(index=False))

    sensor_col = detectar_columna(df, SENSOR_CANDIDATES)
    value_col = detectar_columna(df, PRECIP_CANDIDATES)
    time_col = detectar_columna(df, TIME_CANDIDATES)

    print(f"sensor_col: {sensor_col}")
    if sensor_col is not None:
        sensores = sorted(pd.Series(df[sensor_col].dropna().unique()).astype(str).tolist())
        print(f"unique sensors: {sensores}")

    print(f"precip_col: {value_col}")
    if value_col is not None:
        valores = pd.to_numeric(df[value_col], errors="coerce")
        print(
            "precip numeric range:",
            float(valores.min()),
            float(valores.max()),
        )

    print(f"time_col: {time_col}")
    if time_col is not None:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        diffs = ts.sort_values().diff().dropna()
        print("median diff:", diffs.median())
        print("top diffs:")
        print(diffs.value_counts().head(10).to_string())
        print("ts range:", ts.min(), ts.max())

    print(
        "conclusion: series de lluvia con muestreo tipicamente cada 10 minutos; "
        "se tratan como incrementos por basculada, por lo que el resample a 1h usa sum()"
    )
    return df


def cargar_estacion(path):
    meta = STATION_META[path.name]
    df = pd.read_csv(path)

    sensor_col = detectar_columna(df, SENSOR_CANDIDATES)
    value_col = detectar_columna(df, PRECIP_CANDIDATES)
    time_col = detectar_columna(df, TIME_CANDIDATES)

    if sensor_col is None or value_col is None or time_col is None:
        raise ValueError(f"No pude detectar columnas clave en {path.name}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if meta.get("sensor_qa") is not None:
        df_qa = df[df[sensor_col] == meta["sensor_qa"]].copy()
        print(
            f"{meta['nombre']}: se aislan {len(df_qa)} filas del sensor QA {meta['sensor_qa']} "
            "para no usarlas en el proxy"
        )
        df = df[df[sensor_col] == meta["sensor_principal"]].copy()
        print(
            f"{meta['nombre']}: usando solo sensor {meta['sensor_principal']} ({len(df)} filas)"
        )
    else:
        df = df[df[sensor_col] == meta["sensor_principal"]].copy()
        print(
            f"{meta['nombre']}: usando solo sensor {meta['sensor_principal']} ({len(df)} filas)"
        )

    df = df.drop_duplicates()
    serie = (
        df[[time_col, value_col]]
        .rename(columns={time_col: "timestamp", value_col: "precip_mm"})
        .sort_values("timestamp")
        .set_index("timestamp")["precip_mm"]
    )
    serie = serie.groupby(level=0).sum(min_count=1)

    if serie.empty:
        raise ValueError(f"La serie limpia de {path.name} quedo vacia")

    hourly = serie.resample("1h").sum(min_count=1)
    hourly.name = meta["nombre"]
    return hourly


def calcular_p_basin(series_map):
    df = pd.concat(series_map.values(), axis=1, join="outer")
    min_ts = df.index.min()
    max_ts = df.index.max()
    full_index = pd.date_range(min_ts, max_ts, freq="1h")
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    p_basin = df.mean(axis=1, skipna=True)
    n_estaciones = df.notna().sum(axis=1)
    return df, p_basin, n_estaciones


def calcular_proxy_scs_cn(p_basin):
    p_acc_24h = p_basin.rolling(window=24, min_periods=1).sum()
    validos = p_basin.notna().rolling(window=24, min_periods=1).sum()
    p_acc_24h = p_acc_24h.mask(validos < 12)

    s = (25400 / CN) - 254
    ia = 0.2 * s

    q = pd.Series(np.nan, index=p_basin.index, name="Q_scs_proxy")
    no_luvia = p_acc_24h.notna() & (p_acc_24h <= ia)
    q.loc[no_luvia] = 0.0

    lluvia = p_acc_24h.notna() & (p_acc_24h > ia)
    exceso = p_acc_24h.loc[lluvia] - ia
    q.loc[lluvia] = (exceso**2) / (exceso + s)

    return p_acc_24h, q, s, ia


def verificar_salida(path_out):
    df = pd.read_csv(path_out, parse_dates=["timestamp"])
    print("\n=== Verificacion obligatoria ===")
    print(f"len(df): {len(df)}")
    print(f"rango fechas: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    q = df["Q_scs_proxy"]
    print(f"% filas con Q_scs_proxy NaN: {q.isna().mean() * 100:.2f}%")
    print(f"% filas con Q_scs_proxy == 0: {(q == 0).mean() * 100:.2f}%")

    q_no_nan = q.dropna()
    print(f"Q_scs_proxy min: {q_no_nan.min():.6f}")
    print(f"Q_scs_proxy max: {q_no_nan.max():.6f}")
    print(f"Q_scs_proxy mean: {q_no_nan.mean():.6f}")

    ventana = df[(df["timestamp"] >= "2019-07-01") & (df["timestamp"] <= "2020-12-31 23:00:00")]
    print("\nChequeo del hueco de Metromedellin (jul-2019 -> dic-2020):")
    print(f"filas en ventana: {len(ventana)}")
    print("n_estaciones_disponibles value_counts:")
    print(ventana["n_estaciones_disponibles"].value_counts(dropna=False).sort_index().to_string())
    print(f"min n_estaciones_disponibles en ventana: {ventana['n_estaciones_disponibles'].min()}")
    print(f"max n_estaciones_disponibles en ventana: {ventana['n_estaciones_disponibles'].max()}")
    print(f"filas con n_estaciones_disponibles == 2: {(ventana['n_estaciones_disponibles'] == 2).sum()}")
    print(f"filas con P_basin == 0 en ventana: {(ventana['P_basin'] == 0).sum()}")
    print(f"filas con P_basin NaN en ventana: {ventana['P_basin'].isna().sum()}")


def main():
    print("=== PASO 0 - Inspeccion ===")
    for path in HISTORICO_FILES:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo esperado: {path}")
        inspeccionar_csv(path)

    print("\n=== PASO 1 - Limpieza por estacion ===")
    hourly_series = {}
    for path in HISTORICO_FILES:
        meta = STATION_META[path.name]
        hourly_series[meta["nombre"]] = cargar_estacion(path)

    print("\n=== PASO 2 - P_basin ===")
    merged_hourly, p_basin, n_estaciones = calcular_p_basin(hourly_series)
    print(f"timestamp coverage: {merged_hourly.index.min()} -> {merged_hourly.index.max()}")
    print(f"filas horarias en union: {len(merged_hourly)}")
    print(f"P_basin NaN: {p_basin.isna().sum()}")

    print("\n=== PASO 3 - SCS-CN ===")
    p_acc_24h, q, s, ia = calcular_proxy_scs_cn(p_basin)
    print(f"CN importado desde cuenca_la_honda_params.py: {CN}")
    print(f"S = {s:.6f}")
    print(f"Ia = {ia:.6f}")

    print("\n=== PASO 4 - Guardar ===")
    salida = pd.DataFrame(
        {
            "timestamp": p_basin.index,
            "P_basin": p_basin.values,
            "P_acc_24h": p_acc_24h.values,
            "Q_scs_proxy": q.values,
            "n_estaciones_disponibles": n_estaciones.values,
        }
    )
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
    salida.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")

    verificar_salida(out_path)


if __name__ == "__main__":
    main()