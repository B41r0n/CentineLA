# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Fase 3 (diseño) - Variabilidad espacial de la lluvia entre estaciones IDEAM.

Cuantifica el argumento para el pluviómetro propio de la cuenca alta (README §10.1):
por cada par de estaciones, cuenta los días en que alguna supera el umbral de
escorrentía del proxy (abstracción inicial con CN 81.1, ~11.8 mm en 24 h) y en
cuántos de esos días solo una de las dos lo supera.

Para no confundir variabilidad real con fallas de instrumento (periodos reportados
en 0 mm), solo cuenta días con las 24 h registradas y lluvia > 0 en ambas
estaciones, hasta el corte de evaluación (2026-09-30).

Requiere data/raw/historico_*.csv y data/raw/estaciones_candidatas.csv (no se
versionan; se generan con scripts/01 y scripts/02). No escribe archivos.
"""

import contextlib
import importlib.util
import io
import math
from pathlib import Path

import pandas as pd
from cuenca_la_honda_params import CN

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# Cargar 04_scs_cn_proxy via importlib (nombre con número no es identificador válido)
_proxy_spec = importlib.util.spec_from_file_location(
    "scs_cn_proxy", Path(__file__).resolve().parent / "04_scs_cn_proxy.py"
)
_proxy_mod = importlib.util.module_from_spec(_proxy_spec)
_proxy_spec.loader.exec_module(_proxy_mod)

FECHA_CORTE = pd.Timestamp("2026-09-30 23:00")
IA_MM = 0.2 * (25400 / CN - 254)
LAT_LA_HONDA, LON_LA_HONDA = 6.271, -75.564   # mismo punto que 01_buscar_estaciones.py

ESTACIONES = {
    "0027015290": "historico_0027015290.csv",
    "0027015310": "historico_0027015310.csv",
    "0027015330": "historico_0027015330.csv",
}
PARES = [("0027015290", "0027015330"), ("0027015310", "0027015330")]


def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def cargar_series():
    """Serie horaria por estación con la misma limpieza del proxy (sensor 240)."""
    series = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for codigo, archivo in ESTACIONES.items():
            series[codigo] = _proxy_mod.cargar_estacion(DATA_RAW_DIR / archivo)
    return series


def comparar_par(serie_a, serie_b):
    """Días con 24 h completas y lluvia en ambas; cuántos superan Ia en una sola."""
    par = pd.concat([serie_a, serie_b], axis=1, join="outer")
    par = par[par.index <= FECHA_CORTE]
    diario = par.resample("1D").sum(min_count=24)
    horas = par.notna().resample("1D").sum()
    diario = diario[horas.min(axis=1) == 24]
    ambas = diario[(diario > 0).all(axis=1)]
    supera = ambas > IA_MM
    alguna = supera.any(axis=1)
    una_sola = supera.sum(axis=1) == 1
    return {
        "dias_lluvia_ambas": len(ambas),
        "dias_supera_alguna": int(alguna.sum()),
        "dias_supera_una_sola": int(una_sola.sum()),
        "correlacion_diaria": float(ambas.corr().iloc[0, 1]),
        "rango": (ambas.index.min().date(), ambas.index.max().date()),
    }


def main():
    catalogo = pd.read_csv(DATA_RAW_DIR / "estaciones_candidatas.csv", dtype={"codigo": str})
    catalogo["codigo"] = catalogo["codigo"].str.zfill(10)
    coords = catalogo.set_index("codigo")[["nombre", "lat", "lon"]]
    series = cargar_series()

    print(f"Umbral de escorrentía (Ia con CN {CN}): {IA_MM:.2f} mm en 24 h")
    for codigo in ESTACIONES:
        fila = coords.loc[codigo]
        d = distancia_km(fila["lat"], fila["lon"], LAT_LA_HONDA, LON_LA_HONDA)
        print(f"{fila['nombre']} ({codigo}): {d:.1f} km de La Honda")

    for cod_a, cod_b in PARES:
        a, b = coords.loc[cod_a], coords.loc[cod_b]
        r = comparar_par(series[cod_a], series[cod_b])
        dist = distancia_km(a["lat"], a["lon"], b["lat"], b["lon"])
        pct = r["dias_supera_una_sola"] / r["dias_supera_alguna"]
        print(f"\n=== {a['nombre']} - {b['nombre']} ({dist:.1f} km), {r['rango'][0]} -> {r['rango'][1]}")
        print(f"Días con 24 h completas y lluvia en ambas: {r['dias_lluvia_ambas']}")
        print(f"Días en que alguna supera Ia: {r['dias_supera_alguna']}")
        print(f"... en que solo una lo supera: {r['dias_supera_una_sola']} ({pct:.0%})")
        print(f"Correlación diaria de la lluvia: {r['correlacion_diaria']:.2f}")


if __name__ == "__main__":
    main()
