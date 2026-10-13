# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""Tests para la deduplicación de actualización en vivo."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
ACT_PATH = BASE_DIR / "simulate" / "11_actualizar_ahora.py"


def _load_actualizar_mod():
    spec = importlib.util.spec_from_file_location("actualizar_ahora", ACT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def act(tmp_path):
    mod = _load_actualizar_mod()
    mod.DATA_RAW_DIR = tmp_path
    return mod


def _lote(inicio: str) -> pd.DataFrame:
    """Genera 6 timestamps cada 10 min para los sensores 240 y 257 de Olaya."""
    timestamps = pd.date_range(inicio, periods=6, freq="10min").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    filas = []
    for ts in timestamps:
        for sensor in ("240", "257"):
            filas.append(
                {
                    "fechaobservacion": ts,
                    "codigoestacion": "0027015330",
                    "codigosensor": sensor,
                    "valorobservado": 0.2,
                }
            )
    return pd.DataFrame(filas)


def test_append_sin_duplicados_mantiene_ambos_sensores(act):
    codigo = "0027015330"

    filas_00 = act._append_sin_duplicados(codigo, _lote("2026-10-01 00:00:00"))
    filas_01 = act._append_sin_duplicados(codigo, _lote("2026-10-01 01:00:00"))
    filas_dup = act._append_sin_duplicados(codigo, _lote("2026-10-01 01:00:00"))

    assert filas_00 == 12
    assert filas_01 == 12
    assert filas_dup == 0

    df = pd.read_csv(
        act.DATA_RAW_DIR / f"historico_{codigo}.csv",
        dtype={"codigosensor": str},
    )
    conteo = df["codigosensor"].astype(str).str.lstrip("0").value_counts().to_dict()
    assert conteo == {"240": 12, "257": 12}
