"""
Fase 2, paso 5 - ETL de features para el proxy SCS-CN de La Honda.

Entrada:
    data/processed/proxy_q_la_honda.csv

Salida:
    data/processed/filas_baja_calidad.csv
    data/processed/dataset_6h.csv
    data/processed/dataset_12h.csv
    data/processed/dataset_24h.csv

Uso:
    python 06_etl_features.py
"""

from pathlib import Path
import os

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
QA_PATH = DATA_PROCESSED_DIR / "filas_baja_calidad.csv"

HORIZONTES = (6, 12, 24)
LOW_QUALITY_MIN_ESTACIONES = 2

# Ventanas conocidas documentadas en el contexto tecnico.
PAJARITO_FIN_ACTIVO = pd.Timestamp("2020-03-28 23:59:59")
PAJARITO_SUSPENDIDO_INICIO = pd.Timestamp("2020-03-29 00:00:00")
METROMEDELLIN_GAP_INICIO = pd.Timestamp("2019-02-18 16:00:00")
METROMEDELLIN_GAP_FIN = pd.Timestamp("2019-07-26 11:00:00")


def cargar_proxy():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _mes_intersecta_rango(periodo_mensual, inicio, fin):
    inicio_mes = periodo_mensual.to_timestamp(how="start")
    fin_mes = periodo_mensual.to_timestamp(how="end")
    return (inicio_mes <= fin) and (fin_mes >= inicio)


def diagnosticar_disponibilidad_pre_filtro(df):
    df_diag = df.copy()
    df_diag["timestamp"] = pd.to_datetime(df_diag["timestamp"], errors="coerce")
    df_diag = df_diag.dropna(subset=["timestamp"])
    df_diag["ym"] = df_diag["timestamp"].dt.to_period("M")

    conteos = (
        df_diag[df_diag["n_estaciones_disponibles"].isin([0, 1, 2, 3])]
        .groupby(["ym", "n_estaciones_disponibles"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1, 2, 3], fill_value=0)
        .rename(columns={0: "n0", 1: "n1", 2: "n2", 3: "n3"})
        .astype(int)
    )

    total_mes = df_diag.groupby("ym").size().rename("total_mes").astype(int)
    tabla = conteos.join(total_mes, how="outer").fillna(0)
    tabla[["n0", "n1", "n2", "n3", "total_mes"]] = tabla[
        ["n0", "n1", "n2", "n3", "total_mes"]
    ].astype(int)

    tabla["pct_n0"] = (tabla["n0"] / tabla["total_mes"]).where(tabla["total_mes"] > 0, 0.0)
    tabla["pct_n1"] = (tabla["n1"] / tabla["total_mes"]).where(tabla["total_mes"] > 0, 0.0)
    tabla["pct_nlt2"] = ((tabla["n0"] + tabla["n1"]) / tabla["total_mes"]).where(tabla["total_mes"] > 0, 0.0)

    predominio_nlt2 = tabla[(tabla["pct_n0"] > 0.5) | (tabla["pct_n1"] > 0.5)].copy()

    filas_nlt2 = int((df_diag["n_estaciones_disponibles"] < LOW_QUALITY_MIN_ESTACIONES).sum())

    print("\n=== DIAGNOSTICO PRE-FILTRO (sobre dataframe completo) ===")
    print(f"Filas totales: {len(df_diag)}")
    print(f"Filas con n_estaciones_disponibles < {LOW_QUALITY_MIN_ESTACIONES}: {filas_nlt2}")
    print("\nTabla año-mes x conteos n_estaciones (0,1,2,3):")
    print(tabla[["n0", "n1", "n2", "n3", "total_mes"]].to_string())

    if predominio_nlt2.empty:
        print("\nNo hay meses con predominio (>50%) de n=0 o n=1.")
    else:
        print("\nMeses con predominio (>50%) de n=0 o n=1 (lo que mas se descarta):")
        print(
            predominio_nlt2[["n0", "n1", "n2", "n3", "total_mes", "pct_n0", "pct_n1", "pct_nlt2"]]
            .assign(
                pct_n0=lambda d: (d["pct_n0"] * 100).round(2),
                pct_n1=lambda d: (d["pct_n1"] * 100).round(2),
                pct_nlt2=lambda d: (d["pct_nlt2"] * 100).round(2),
            )
            .to_string()
        )

    meses_con_nlt2 = tabla[(tabla["n0"] + tabla["n1"]) > 0].copy()
    meses_fuera_ventanas = []
    for periodo in meses_con_nlt2.index:
        cubierto_gap_metro = _mes_intersecta_rango(periodo, METROMEDELLIN_GAP_INICIO, METROMEDELLIN_GAP_FIN)
        cubierto_pajarito_suspendido = _mes_intersecta_rango(
            periodo,
            PAJARITO_SUSPENDIDO_INICIO,
            df_diag["timestamp"].max(),
        )

        if not (cubierto_gap_metro or cubierto_pajarito_suspendido):
            meses_fuera_ventanas.append(periodo)

    print("\nCruce con ventanas conocidas:")
    print(
        f"- Pajarito activo hasta: {PAJARITO_FIN_ACTIVO} (suspendido desde {PAJARITO_SUSPENDIDO_INICIO})"
    )
    print(f"- Hueco Metromedellin: {METROMEDELLIN_GAP_INICIO} -> {METROMEDELLIN_GAP_FIN}")

    if not meses_fuera_ventanas:
        print("No se detectan meses con n<2 fuera de las ventanas conocidas.")
    else:
        print("HALLAZGO NUEVO: meses con presencia de n<2 fuera de ventanas conocidas:")
        tabla_hallazgo = tabla.loc[meses_fuera_ventanas, ["n0", "n1", "n2", "n3", "total_mes"]].copy()
        tabla_hallazgo["pct_nlt2"] = (
            ((tabla_hallazgo["n0"] + tabla_hallazgo["n1"]) / tabla_hallazgo["total_mes"]) * 100
        ).round(2)
        print(tabla_hallazgo.to_string())


def clasificar_meses_temporada(fechas):
    periodos = pd.Index(fechas.dropna().dt.to_period("M").unique()).sort_values()
    lluviosa = 0
    seca = 0

    for periodo in periodos:
        if periodo.month in (4, 5, 10, 11):
            lluviosa += 1
        else:
            seca += 1

    return lluviosa, seca, list(periodos.astype(str))


def filtrar_calidad(df):
    filas_baja_calidad = df[df["n_estaciones_disponibles"] < LOW_QUALITY_MIN_ESTACIONES].copy()
    filas_calidad = df[df["n_estaciones_disponibles"] >= LOW_QUALITY_MIN_ESTACIONES].copy()

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    filas_baja_calidad.to_csv(QA_PATH, index=False)

    print("=== PASO 1 - Filtro de calidad ===")
    print(f"Filas totales de entrada: {len(df)}")
    print(f"Filas baja calidad (<{LOW_QUALITY_MIN_ESTACIONES} estaciones): {len(filas_baja_calidad)}")
    print(f"Filas que siguen al ETL: {len(filas_calidad)}")
    print(f"Guardado QA: {QA_PATH}")

    return filas_calidad, filas_baja_calidad


def construir_base_horaria(df):
    base = df[["timestamp", "P_basin", "P_acc_24h", "Q_scs_proxy"]].copy()
    base = base.drop_duplicates(subset=["timestamp"], keep="last")
    base = base.sort_values("timestamp").set_index("timestamp")

    full_index = pd.date_range(base.index.min(), base.index.max(), freq="1h")
    base = base.reindex(full_index)
    base.index.name = "timestamp"

    return base


def diagnostico_etapas_6h(df_proxy, df_calidad, base_horaria):
    total_original = len(df_proxy)
    tras_calidad = len(df_calidad)
    features = armar_features(base_horaria)
    target = base_horaria["Q_scs_proxy"].shift(-6)

    tras_dropna_features = len(features.dropna(subset=features.columns))

    columnas_entrada = list(features.columns)
    conjunto_antes_target = features.copy()
    conjunto_antes_target["target_6h"] = target
    tras_dropna_target = len(conjunto_antes_target.dropna(subset=columnas_entrada + ["target_6h"]))

    print("\n=== DIAGNOSTICO 6h POR ETAPAS ===")
    print(f"Total original: {total_original} | perdido en etapa: 0")
    print(
        f"Tras filtro n_estaciones>=2: {tras_calidad} | perdido en etapa: {total_original - tras_calidad}"
    )
    print(
        f"Tras dropna de features de entrada: {tras_dropna_features} | perdido en etapa: {tras_calidad - tras_dropna_features}"
    )
    print(
        f"Tras dropna de target_6h: {tras_dropna_target} | perdido en etapa: {tras_dropna_features - tras_dropna_target}"
    )


def armar_features(base_horaria):
    features = pd.DataFrame(index=base_horaria.index)
    features["P_basin"] = base_horaria["P_basin"]
    features["Q_actual"] = base_horaria["Q_scs_proxy"]

    for horas in (1, 3, 6, 12, 24):
        features[f"lag_{horas}h"] = base_horaria["P_basin"].shift(horas)

    for horas in (1, 3, 6):
        features[f"Q_lag_{horas}h"] = base_horaria["Q_scs_proxy"].shift(horas)

    for horas in (3, 6, 12, 24, 48):
        features[f"roll_sum_{horas}h"] = base_horaria["P_basin"].rolling(window=horas, min_periods=1).sum()

    features["roll_max_6h"] = base_horaria["P_basin"].rolling(window=6, min_periods=1).max()
    features["hora_dia"] = features.index.hour
    features["mes"] = features.index.month

    return features


def obtener_corte_split_existente(horizonte_horas):
    path_existente = DATA_PROCESSED_DIR / f"dataset_{horizonte_horas}h.csv"
    if not path_existente.exists():
        return None

    try:
        previo = pd.read_csv(path_existente, parse_dates=["timestamp"])
    except Exception as exc:
        print(f"ADVERTENCIA: no pude leer split existente en {path_existente}: {exc}")
        return None

    if "split" not in previo.columns or "timestamp" not in previo.columns:
        print(f"ADVERTENCIA: archivo existente sin columnas split/timestamp: {path_existente}")
        return None

    previo = previo.sort_values("timestamp").reset_index(drop=True)
    test_prev = previo[previo["split"] == "test"]
    if test_prev.empty:
        print(f"ADVERTENCIA: archivo existente no tiene filas test: {path_existente}")
        return None

    return test_prev["timestamp"].min()


def armar_dataset(base_horaria, horizonte_horas):
    target_col = f"target_{horizonte_horas}h"
    dataset = armar_features(base_horaria)
    dataset[target_col] = base_horaria["Q_scs_proxy"].shift(-horizonte_horas)

    columnas_entrada = [c for c in dataset.columns if c != target_col]
    dataset = dataset.dropna(subset=columnas_entrada + [target_col]).copy()
    dataset = dataset.reset_index().rename(columns={"index": "timestamp"})

    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    dataset["split"] = "train"

    corte_ts_existente = obtener_corte_split_existente(horizonte_horas)
    if corte_ts_existente is not None:
        dataset.loc[dataset["timestamp"] >= corte_ts_existente, "split"] = "test"
        print(
            f"Split preservado desde archivo existente (h={horizonte_horas}): "
            f"corte_ts={corte_ts_existente}"
        )
    else:
        n_total = len(dataset)
        corte = int(n_total * 0.82)
        if n_total > 0:
            corte = max(1, min(corte, n_total - 1)) if n_total > 1 else 1
        dataset.loc[corte:, "split"] = "test"
        print(f"Split cronologico nuevo por proporcion 82/18 (h={horizonte_horas}).")

    return dataset, target_col


def resumir_dataset(df, target_col):
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    fechas_train = (train["timestamp"].min(), train["timestamp"].max()) if not train.empty else (pd.NaT, pd.NaT)
    fechas_test = (test["timestamp"].min(), test["timestamp"].max()) if not test.empty else (pd.NaT, pd.NaT)

    total_na_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100 if len(df) else 0.0
    target_zero_pct = df[target_col].eq(0).mean() * 100 if len(df) else 0.0

    print(f"n filas train: {len(train)}")
    print(f"n filas test: {len(test)}")
    print(f"rango train: {fechas_train[0]} -> {fechas_train[1]}")
    print(f"rango test: {fechas_test[0]} -> {fechas_test[1]}")
    print(f"% target==0: {target_zero_pct:.2f}")
    print(f"% NaN restante: {total_na_pct:.2f}")
    if not train.empty and not test.empty:
        print(f"max(fecha train) < min(fecha test): {train['timestamp'].max() < test['timestamp'].min()}")


def resumir_temporada_split(df):
    for split_nombre in ("train", "test"):
        subconjunto = df[df["split"] == split_nombre].copy()
        if subconjunto.empty:
            print(f"{split_nombre}: sin filas")
            continue

        lluviosa, seca, meses = clasificar_meses_temporada(subconjunto["timestamp"])
        print(
            f"{split_nombre}: rango {subconjunto['timestamp'].min()} -> {subconjunto['timestamp'].max()}"
        )
        print(f"{split_nombre}: meses unicos en rango: {len(meses)}")
        print(f"{split_nombre}: meses lluviosos (abr-may, oct-nov): {lluviosa}")
        print(f"{split_nombre}: meses secos (resto): {seca}")


def verificar_archivo_guardado(path, target_col):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"\nVerificacion lectura real: {path}")
    print(f"len(df): {len(df)}")
    resumir_dataset(df, target_col)
    resumir_temporada_split(df)
    return df


def procesar_horizonte(base_horaria, horizonte_horas):
    print(f"\n=== PASO 2-5 - Horizonte {horizonte_horas}h ===")
    target_col = f"target_{horizonte_horas}h"
    dataset, target_col = armar_dataset(base_horaria, horizonte_horas)

    if dataset.empty:
        raise ValueError(f"El dataset de {horizonte_horas}h quedo vacio")

    nombre_archivo = DATA_PROCESSED_DIR / f"dataset_{horizonte_horas}h.csv"
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    dataset.to_csv(nombre_archivo, index=False)
    print(f"Guardado: {nombre_archivo}")

    verificiado = verificar_archivo_guardado(nombre_archivo, target_col)

    return verificiado


def verificar_dataset_6h(df):
    target_col = "target_6h"
    target_series = df[target_col].dropna()
    p90 = target_series.quantile(0.90)
    train_extremos = df[(df["split"] == "train") & (df[target_col] > p90)]
    test_extremos = df[(df["split"] == "test") & (df[target_col] > p90)]

    print("\n=== PASO 6 - Extra dataset_6h ===")
    print(f"percentil 90 de target_6h: {p90:.6f}")
    print(f"filas target_6h > p90 en train: {len(train_extremos)}")
    print(f"filas target_6h > p90 en test: {len(test_extremos)}")


def main():
    proxy = cargar_proxy()
    diagnosticar_disponibilidad_pre_filtro(proxy)
    filas_calidad, _filas_baja = filtrar_calidad(proxy)
    base_horaria = construir_base_horaria(filas_calidad)

    diagnostico_etapas_6h(proxy, filas_calidad, base_horaria)

    datasets = {}
    for horizonte in HORIZONTES:
        try:
            datasets[horizonte] = procesar_horizonte(base_horaria, horizonte)
        except Exception as exc:
            if horizonte == 6:
                raise
            print(f"ADVERTENCIA: no pude generar dataset_{horizonte}h.csv sin bloquear el pipeline: {exc}")

    if 6 in datasets:
        verificar_dataset_6h(datasets[6])


if __name__ == "__main__":
    main()