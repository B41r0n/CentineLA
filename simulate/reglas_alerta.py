# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
Reglas del sistema de alertas — única fuente para el dashboard, el gateway
simulado (09) y las métricas (10).

1. Escala ternaria sobre la probabilidad: NORMAL < 0.30 ≤ PRECAUCIÓN < 0.70 ≤ ALERTA
   (criterio proba >= umbral).

2. Regla de lluvia seca: si no llovió nada en las últimas 24 h, la probabilidad
   no puede superar la frecuencia histórica de riesgo en esas horas.
   Sin lluvia previa, para que haya escorrentía a 6 h tendrían que caer ~12 mm
   nuevos (la abstracción inicial con CN 81.1), algo que el modelo no puede
   anticipar con las features disponibles. Sin esta regla, el clasificador se
   apoya en hora_dia/mes y a veces marca PRECAUCIÓN o ALERTA con 0 mm: en la
   ventana de prueba fueron 89 avisos en horas secas y ninguno acertó.
   PROBA_MAX_SIN_LLUVIA = 68 / 10 303: horas de riesgo entre las horas con 0 mm
   en 24 h del set de entrenamiento (corte 2025-07-06 16:00).
"""

import numpy as np

UMBRAL_PRECAUCION = 0.30
UMBRAL_ALERTA = 0.70

LLUVIA_SECA_24H_MM = 0.0
PROBA_MAX_SIN_LLUVIA = 0.0066


def clasificar_estado(proba: float) -> str:
    """Sistema ternario de alertas: NORMAL / PRECAUCIÓN / ALERTA."""
    if proba >= UMBRAL_ALERTA:
        return "ALERTA"
    elif proba >= UMBRAL_PRECAUCION:
        return "PRECAUCIÓN"
    return "NORMAL"


def ajustar_proba(proba, roll_sum_24h):
    """Aplica la regla de lluvia seca a la probabilidad del modelo.

    Acepta escalares o arreglos. Un roll_sum_24h NaN no activa la regla.
    Es idempotente: aplicarla dos veces da lo mismo.
    """
    p = np.asarray(proba, dtype=float)
    seco = np.asarray(roll_sum_24h, dtype=float) <= LLUVIA_SECA_24H_MM
    out = np.where(seco, np.minimum(p, PROBA_MAX_SIN_LLUVIA), p)
    return float(out) if out.ndim == 0 else out
