# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Módulo 11: Actualización en vivo desde IDEAM (Socrata) con BD SQLite local.

Flujo:
1. Para cada estación, lee el último timestamp en data/raw/historico_<codigo>.csv
2. Consulta Socrata (s54a-sgyg) solo el rango [último_ts, ahora]
3. Append al CSV sin duplicar por (fechaobservacion, codigoestacion)
4. Ejecuta pipeline: 04_scs_cn_proxy.py -> 06_etl_features.py -> 09_gateway_simulado.py
5. Regenera data/processed/centinela.db (tabla lecturas) desde log_gateway_simulado.csv
6. Siembra la BD al importar el módulo si no existe (lazy)
"""

import importlib.util
import os
import sys
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sodapy import Socrata

# Cargar 02_pull_historico via importlib (nombre con número no es identificador válido)
_PULL_HIST_PATH = Path(__file__).resolve().parent.parent / "scripts" / "02_pull_historico.py"
_pull_spec = importlib.util.spec_from_file_location("pull_historico", _PULL_HIST_PATH)
_pull_mod = importlib.util.module_from_spec(_pull_spec)
_pull_spec.loader.exec_module(_pull_mod)

CAMPOS_CODIGO_POSIBLES = _pull_mod.CAMPOS_CODIGO_POSIBLES
DATASET_HISTORICO = _pull_mod.DATASET_HISTORICO
candidatos_codigo = _pull_mod.candidatos_codigo
detecta_campo_codigo = _pull_mod.detecta_campo_codigo
inspecciona = _pull_mod.inspecciona
normaliza_texto = _pull_mod.normaliza_texto
obtener_meta_catalogo = _pull_mod.obtener_meta_catalogo

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = DATA_PROCESSED_DIR / "centinela.db"

ESTACIONES_DEFAULT = ("0027015310", "0027015330")  # Metromedellin, Olaya Herrera
FECHA_INICIO_DEFAULT = "2026-09-30T18:00:00"  # día después del último dato conocido


def _partes_dominio():
    """Devuelve las partes del dominio ofuscado (misma técnica que 02_pull_historico)."""
    return [chr(119) * 2 + chr(119), "datos", "gov", "co"]


def _get_client(username: str, password: str, app_token: str | None = None) -> Socrata:
    """Crea cliente Socrata con el dominio ofuscado."""
    return Socrata(
        ".".join(_partes_dominio()),
        app_token,
        username=username,
        password=password,
        timeout=120,
    )


def _leer_ultimo_ts(codigo: str) -> str:
    """Lee el último fechaobservacion del CSV histórico de la estación.
    Si no existe, usa (último timestamp del proxy - 72 h) para no perder el histórico.
    Si el proxy tampoco existe, devuelve FECHA_INICIO_DEFAULT."""
    path = DATA_RAW_DIR / f"historico_{codigo}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, usecols=["fechaobservacion"])
            if df.empty:
                return FECHA_INICIO_DEFAULT
            df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
            df = df.dropna(subset=["fechaobservacion"])
            if df.empty:
                return FECHA_INICIO_DEFAULT
            return df["fechaobservacion"].max().strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            return FECHA_INICIO_DEFAULT

    # Fallback: punto de partida seguro desde el proxy existente
    proxy_path = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
    if proxy_path.exists():
        try:
            proxy = pd.read_csv(proxy_path, parse_dates=["timestamp"])
            if not proxy.empty:
                ts = proxy["timestamp"].max() - timedelta(hours=72)
                # Redondear a la hora
                ts = ts.floor("h")
                return ts.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

    return FECHA_INICIO_DEFAULT


def _append_sin_duplicados(codigo: str, df_nuevo: pd.DataFrame) -> int:
    """Append al CSV histórico evitando duplicados por (fechaobservacion, codigoestacion).
    Devuelve número de filas nuevas añadidas."""
    path = DATA_RAW_DIR / f"historico_{codigo}.csv"
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_existente = pd.read_csv(path)
        # Normalizar columnas de fecha
        if "fechaobservacion" in df_existente.columns:
            df_existente["fechaobservacion"] = pd.to_datetime(
                df_existente["fechaobservacion"], errors="coerce"
            )
        if "fechaobservacion" in df_nuevo.columns:
            df_nuevo["fechaobservacion"] = pd.to_datetime(
                df_nuevo["fechaobservacion"], errors="coerce"
            )

        # Key de deduplicación
        key_cols = [c for c in ["fechaobservacion", "codigoestacion"] if c in df_existente.columns and c in df_nuevo.columns]
        if key_cols:
            df_combined = pd.concat([df_existente, df_nuevo], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=key_cols, keep="last")
            filas_nuevas = len(df_combined) - len(df_existente)
            df_combined.to_csv(path, index=False)
            return filas_nuevas
        else:
            # Fallback: append todo
            df_nuevo.to_csv(path, mode="a", header=False, index=False)
            return len(df_nuevo)
    else:
        df_nuevo.to_csv(path, index=False)
        return len(df_nuevo)


def _consultar_paginado(client, dataset_id: str, where: str, page_size: int = 5000) -> list:
    """Consulta Socrata paginando por offset hasta que una página traiga < page_size filas."""
    resultados = []
    offset = 0
    while True:
        pagina = client.get(
            dataset_id,
            where=where,
            order=":id",
            limit=page_size,
            offset=offset,
        )
        if not pagina:
            break
        resultados.extend(pagina)
        if len(pagina) < page_size:
            break
        offset += page_size
    return resultados


def _fusionar_proxy(proxy_old: pd.DataFrame, proxy_new: pd.DataFrame) -> pd.DataFrame:
    """Fusiona proxy histórico (old) con el recién generado (new).
    Si new es una regeneración completa (min muy antiguo), conserva old hasta
    48 h antes de su máximo y usa new a partir de ahí. Si new es solo el slice
    nuevo (caso producción con raw vacío), conserva old hasta 24 h después del
    inicio de new."""
    if proxy_old.empty:
        return proxy_new.copy()
    if proxy_new.empty:
        return proxy_old.copy()

    corte_nuevo = proxy_new["timestamp"].min() + timedelta(hours=24)
    corte_viejo = proxy_old["timestamp"].max() - timedelta(hours=48)
    corte = max(corte_nuevo, corte_viejo)

    old_part = proxy_old[proxy_old["timestamp"] < corte].copy()
    new_part = proxy_new[proxy_new["timestamp"] >= corte].copy()

    merged = pd.concat([old_part, new_part], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged


def _ejecutar_script(script_relativo: str) -> tuple[bool, str]:
    """Ejecuta un script del pipeline como subprocess.
    Devuelve (ok, output/error)."""
    script_path = BASE_DIR / script_relativo
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return False, f"{script_relativo} falló (code {result.returncode}): {result.stderr[:500]}"
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, f"{script_relativo} timeout (>120s)"
    except Exception as e:
        return False, f"{script_relativo} error: {e}"


def _regenerar_bd_sqlite() -> None:
    """Regenera data/processed/centinela.db tabla 'lecturas' desde log_gateway_simulado.csv."""
    LOG_PATH = DATA_PROCESSED_DIR / "log_gateway_simulado.csv"
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No existe {LOG_PATH}")

    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    # Añadir columna actualizado_en
    df["actualizado_en"] = pd.Timestamp.now()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # Reemplazar contenido completo (DROP + CREATE)
        df.to_sql("lecturas", conn, if_exists="replace", index=False)
        # Índice para consultas por timestamp
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lecturas_ts ON lecturas(timestamp)")


def _asegurar_db_sembrada() -> None:
    """Si la BD no existe, la siembra desde el log actual del repo."""
    if not DB_PATH.exists():
        _regenerar_bd_sqlite()


# Ejecutar siembra perezosa al importar
_asegurar_db_sembrada()


def actualizar_ahora(estaciones: tuple[str, ...] = ESTACIONES_DEFAULT) -> dict[str, Any]:
    """Ejecuta el flujo completo de actualización en vivo.
    Devuelve dict con ok=True/False y detalles o error.
    """
    inicio = time.time()

    # 1. Credenciales
    username = os.environ.get("SODAPY_USERNAME")
    password = os.environ.get("SODAPY_PASSWORD")
    app_token = os.environ.get("SODAPY_APP_TOKEN")

    if not username or not password:
        return {
            "ok": False,
            "error": "Faltan credenciales SODAPY_USERNAME/SODAPY_PASSWORD",
        }

    # 2. Cliente Socrata
    try:
        client = _get_client(username, password, app_token)
    except Exception as e:
        return {"ok": False, "error": f"Error creando cliente Socrata: {e}"}

    # 3. Detectar campo de código (reusando lógica de 02)
    try:
        cols = inspecciona(client, DATASET_HISTORICO)
        campo_codigo = detecta_campo_codigo(cols)
        if campo_codigo is None:
            return {"ok": False, "error": "No se detectó campo de código de estación en dataset histórico"}
    except Exception as e:
        return {"ok": False, "error": f"Error inspeccionando dataset: {e}"}

    # 4. Para cada estación, consultar rango [último_ts, ahora] con paginación
    filas_nuevas_por_estacion = {}
    ultima_lectura_por_estacion = {}
    ahora = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%dT%H:%M:%S")

    for codigo in estaciones:
        try:
            ultimo_ts = _leer_ultimo_ts(codigo)
            if ultimo_ts >= ahora:
                filas_nuevas_por_estacion[codigo] = 0
                ultima_lectura_por_estacion[codigo] = ultimo_ts
                continue

            # Construir where clause con candidatos de código
            where_base = " or ".join(
                f"{campo_codigo}='{cand}'" for cand in candidatos_codigo(codigo)
            )
            where_rango = (
                f"({where_base}) and fechaobservacion between '{ultimo_ts}' and '{ahora}'"
            )

            resultados = _consultar_paginado(client, DATASET_HISTORICO, where_rango)

            if resultados:
                df_nuevo = pd.DataFrame.from_records(resultados)
                n_nuevas = _append_sin_duplicados(codigo, df_nuevo)
                filas_nuevas_por_estacion[codigo] = n_nuevas
                # Última fechaobservacion del CSV resultante
                df_csv = pd.read_csv(DATA_RAW_DIR / f"historico_{codigo}.csv")
                ult_ts = pd.to_datetime(df_csv["fechaobservacion"], errors="coerce").max()
                ultima_lectura_por_estacion[codigo] = (
                    ult_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ult_ts) else ""
                )
            else:
                filas_nuevas_por_estacion[codigo] = 0
                ultima_lectura_por_estacion[codigo] = ""

        except Exception as e:
            return {"ok": False, "error": f"Error consultando estación {codigo}: {e}"}

    # 5. Guardar proxy anterior en memoria antes de que 04 lo sobrescriba
    PROXY_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
    proxy_old = pd.DataFrame()
    if PROXY_PATH.exists():
        try:
            proxy_old = pd.read_csv(PROXY_PATH, parse_dates=["timestamp"])
        except Exception:
            proxy_old = pd.DataFrame()

    # 6. Ejecutar pipeline en orden
    ok, out = _ejecutar_script("simulate/04_scs_cn_proxy.py")
    if not ok:
        return {"ok": False, "error": out}

    # Fusionar proxy para no perder el histórico
    if not proxy_old.empty and PROXY_PATH.exists():
        try:
            proxy_new = pd.read_csv(PROXY_PATH, parse_dates=["timestamp"])
            proxy_merged = _fusionar_proxy(proxy_old, proxy_new)
            proxy_merged.to_csv(PROXY_PATH, index=False)
        except Exception as e:
            return {"ok": False, "error": f"Error fusionando proxy: {e}"}

    ok, out = _ejecutar_script("simulate/06_etl_features.py")
    if not ok:
        return {"ok": False, "error": out}

    ok, out = _ejecutar_script("simulate/09_gateway_simulado.py")
    if not ok:
        return {"ok": False, "error": out}

    # 7. Regenerar BD SQLite
    try:
        _regenerar_bd_sqlite()
    except Exception as e:
        return {"ok": False, "error": f"Error regenerando BD SQLite: {e}"}

    # 8. Obtener última lectura del log regenerado
    LOG_PATH = DATA_PROCESSED_DIR / "log_gateway_simulado.csv"
    ultima_lectura = ""
    if LOG_PATH.exists():
        df_log = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
        if not df_log.empty:
            ultima_lectura = df_log["timestamp"].max().strftime("%Y-%m-%d %H:%M")

    return {
        "ok": True,
        "filas_nuevas_por_estacion": filas_nuevas_por_estacion,
        "ultima_lectura_por_estacion": ultima_lectura_por_estacion,
        "ultima_lectura": ultima_lectura,
        "segundos": round(time.time() - inicio, 1),
    }