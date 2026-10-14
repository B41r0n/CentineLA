# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""Tests de reglas_alerta.py: escala ternaria y regla de lluvia seca."""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def reglas():
    spec = importlib.util.spec_from_file_location("reglas_alerta", BASE_DIR / "simulate" / "reglas_alerta.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regla_lluvia_seca_escalar(reglas):
    assert reglas.ajustar_proba(0.95, 0.0) == pytest.approx(reglas.PROBA_MAX_SIN_LLUVIA)
    assert reglas.ajustar_proba(0.001, 0.0) == pytest.approx(0.001)  # nunca sube la probabilidad
    assert reglas.ajustar_proba(0.95, 0.1) == pytest.approx(0.95)  # con algo de lluvia no actúa
    assert reglas.ajustar_proba(0.95, float("nan")) == pytest.approx(0.95)  # sin dato no actúa


def test_regla_lluvia_seca_arreglo_e_idempotente(reglas):
    p = np.array([0.9, 0.5, 0.9])
    r24 = np.array([0.0, 0.0, 3.0])
    una = reglas.ajustar_proba(p, r24)
    assert una[0] == una[1] == reglas.PROBA_MAX_SIN_LLUVIA and una[2] == 0.9
    np.testing.assert_array_equal(reglas.ajustar_proba(una, r24), una)


def test_tope_por_debajo_de_precaucion(reglas):
    assert reglas.clasificar_estado(reglas.PROBA_MAX_SIN_LLUVIA) == "NORMAL"


def test_gateway_y_metricas_usan_el_modulo_compartido():
    # Evita que vuelvan a divergir los umbrales (como pasó con > vs >=).
    for script in ("09_gateway_simulado.py", "10_metricas_ternario.py"):
        src = (BASE_DIR / "simulate" / script).read_text(encoding="utf-8")
        assert "from reglas_alerta import" in src, script
        assert not re.search(r"^UMBRAL_(PRECAUCION|ALERTA)\s*=", src, re.MULTILINE), script
