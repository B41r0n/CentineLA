# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Fase 1, paso 1 — Identificar estaciones IDEAM cercanas a Q. La Honda / Aranjuez.

Corre en tu máquina (el sandbox de Claude no tiene salida a datos.gov.co).

Uso:
    pip install sodapy pandas
    python 01_buscar_estaciones.py

Salida:
    ../data/raw/estaciones_candidatas.csv  — rankeadas por distancia a La Honda
"""

import math
from pathlib import Path
import pandas as pd
from sodapy import Socrata

# --- Referencia: Q. La Honda / barrio Sevilla, Comuna 4 Aranjuez, Medellín ---
LAT_LA_HONDA = 6.271
LON_LA_HONDA = -75.564

# Municipios del Valle de Aburrá — cualquier estación acá es geográficamente
# relevante para lluvia/clima que afecta a Medellín (el radar/pluviometría
# no respeta límites municipales a esta escala).
MUNICIPIOS_VALLE_ABURRA = {
    "medellin", "medellín", "bello", "itagui", "itagüi", "itagüí",
    "envigado", "sabaneta", "la estrella", "caldas", "copacabana",
    "girardota", "barbosa",
}

DATASET_CATALOGO = "hp9r-jxuu"  # Catálogo Nacional de Estaciones del IDEAM
DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def normaliza(s):
    if s is None:
        return ""
    return str(s).strip().lower()


def main():
    print("Conectando a datos.gov.co (Socrata, sin app_token — puede ir lento/limitado)...")
    client = Socrata("www.datos.gov.co", None, timeout=60)

    # Traemos Antioquia completo (~cientos de filas, no miles) y filtramos local.
    results = client.get(
        DATASET_CATALOGO,
        Departamento="Antioquia",
        limit=5000,
    )
    df = pd.DataFrame.from_records(results)
    print(f"Estaciones en Antioquia (catálogo IDEAM): {len(df)}")

    if df.empty:
        print("No llegaron datos. Revisá conexión o nombre del dataset.")
        return

    # El campo de ubicación puede venir anidado (latitude/longitude) o como
    # columnas planas LATITUD/LONGITUD según la versión de la API. Cubrimos
    # ambos casos.
    def get_lat(row):
        if "latitud" in row and pd.notna(row.get("latitud")):
            return float(row["latitud"])
        if "ubicaci_n" in row and isinstance(row["ubicaci_n"], dict):
            return float(row["ubicaci_n"].get("latitude", "nan"))
        return float("nan")

    def get_lon(row):
        if "longitud" in row and pd.notna(row.get("longitud")):
            return float(row["longitud"])
        if "ubicaci_n" in row and isinstance(row["ubicaci_n"], dict):
            return float(row["ubicaci_n"].get("longitude", "nan"))
        return float("nan")

    df["lat"] = df.apply(get_lat, axis=1)
    df["lon"] = df.apply(get_lon, axis=1)
    df = df.dropna(subset=["lat", "lon"])

    # Filtro Valle de Aburrá (columna Municipio puede venir como "municipio")
    col_municipio = "municipio" if "municipio" in df.columns else "Municipio"
    df["municipio_norm"] = df[col_municipio].apply(normaliza)
    df_valle = df[df["municipio_norm"].isin(MUNICIPIOS_VALLE_ABURRA)].copy()
    print(f"Estaciones en Valle de Aburrá: {len(df_valle)}")

    df_valle["distancia_km"] = df_valle.apply(
        lambda r: haversine_km(LAT_LA_HONDA, LON_LA_HONDA, r["lat"], r["lon"]), axis=1
    )

    col_estado = "estado" if "estado" in df_valle.columns else "Estado"
    col_categoria = "categoria" if "categoria" in df_valle.columns else "Categoria"
    col_codigo = "codigo" if "codigo" in df_valle.columns else "Codigo"
    col_nombre = "nombre" if "nombre" in df_valle.columns else "Nombre"

    cols_out = [col_codigo, col_nombre, col_categoria, col_estado, col_municipio, "distancia_km", "lat", "lon"]
    cols_out = [c for c in cols_out if c in df_valle.columns]

    out = df_valle.sort_values("distancia_km")[cols_out]

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_RAW_DIR / "estaciones_candidatas.csv"
    out.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}\n")
    print("Top 15 más cercanas a La Honda:")
    print(out.head(15).to_string(index=False))

    print(
        "\nSiguiente paso manual: elegí el Codigo de la estación (idealmente "
        "Activa, Pluviométrica/Pluviográfica/Climatológica, con telemetría "
        "si hay opción) y pasalo a 02_pull_historico.py"
    )


if __name__ == "__main__":
    main()
