# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "dashboard"))
import matplotlib
matplotlib.use("Agg")
import importlib
import app
importlib.reload(app)
import pandas as pd
import numpy as np

proxy = pd.read_csv(BASE_DIR / "data" / "processed" / "proxy_q_la_honda.csv", parse_dates=["timestamp"])
proxy = proxy.sort_values("timestamp").reset_index(drop=True)
import joblib
modelo = joblib.load(BASE_DIR / "simulate" / "models" / "clf_6h.joblib")

print("=== FIX 1 ===")
ult_24h, ult_30d, anio_mm, ref_fecha = app._lluvia_acumulada(proxy)
print(f"  ref_fecha (de proxy.max, NO del reloj): {ref_fecha}")
print(f"  ult_24h = {ult_24h:.3f}  (0.0 = genuinamente sin lluvia el ultimo dia)")
print(f"  ult_30d = {ult_30d:.3f}  (no 0 -> ventana de 30d captura algo de lluvia)")
print(f"  anio    = {anio_mm:.3f}  (acumulado del anio en curso)")
assert isinstance(ref_fecha, str) and len(ref_fecha) == 10 and ref_fecha[4] == "-" and ref_fecha[7] == "-", \
    f"ref_fecha debe ser YYYY-MM-DD, se obtuvo: {ref_fecha}"
assert ref_fecha == str(proxy["timestamp"].max().date()), \
    "ref_fecha debe coincidir con proxy.max(), no con una fecha fija"
assert ult_24h >= 0 and ult_30d >= 0 and anio_mm > 0, "año debe ser >0"
print("  OK: rolling windows correctos, ref_fecha es proxy.max(), anio > 0")

print("\n=== FIX 2 ===")
import inspect
src = inspect.getsource(app._mapa_folium)
assert "fit_bounds" in src, "debe usar fit_bounds"
assert "pad = 0.015" in src, "debe tener padding"
print("  fit_bounds presente en _mapa_folium: OK")
print("  padding definido: OK")

print("\n=== FEATURE simulador ===")
assert hasattr(app, "_simulador_modelo"), "_simulador_modelo debe existir"
assert hasattr(app, "_ESCENARIOS"), "_ESCENARIOS debe existir"
assert len(app._ESCENARIOS) == 3, "deben ser 3 escenarios"
print("  _simulador_modelo y _ESCENARIOS presentes: OK")

escenario_lluvia = list(app._ESCENARIOS.values())[0]
x_test = {
    "P_basin":       escenario_lluvia["sim_lag_1h"],
    "Q_actual":      escenario_lluvia["sim_Q"],
    "lag_1h":        escenario_lluvia["sim_lag_1h"],
    "lag_3h":        escenario_lluvia["sim_lag_3h"],
    "lag_6h":        escenario_lluvia["sim_lag_6h"],
    "lag_12h":       escenario_lluvia["sim_lag_6h"] * 0.5,
    "lag_24h":       escenario_lluvia["sim_lag_6h"] * 0.25,
    "Q_lag_1h":      escenario_lluvia["sim_Q"],
    "Q_lag_3h":      escenario_lluvia["sim_Q"],
    "Q_lag_6h":      escenario_lluvia["sim_Q"],
    "roll_sum_3h":   (escenario_lluvia["sim_lag_1h"] + escenario_lluvia["sim_lag_3h"]) * 0.5,
    "roll_sum_6h":   escenario_lluvia["sim_lag_6h"],
    "roll_sum_12h":  escenario_lluvia["sim_roll12"],
    "roll_sum_24h":  escenario_lluvia["sim_roll24"],
    "roll_sum_48h":  escenario_lluvia["sim_roll48"],
    "roll_max_6h":   escenario_lluvia["sim_lag_1h"],
    "hora_dia":      14.0,
    "mes":           7.0,
}
x_df = pd.DataFrame([x_test])[app.FEATURE_COLS]
proba = float(modelo.predict_proba(x_df)[0, 1])
estado = "ALERTA_6H" if proba > app.UMBRAL_ALERTA else "NORMAL"
print(f"  Escenario lluvia fuerte: proba={proba:.4f}  estado={estado}")
assert 0 <= proba <= 1, "proba debe estar en [0,1]"

x_normal = {k: 0.0 for k in x_test}
x_normal["hora_dia"] = 14.0
x_normal["mes"] = 7.0
x_normal["Q_actual"] = 0.0
x_normal["Q_lag_1h"] = x_normal["Q_lag_3h"] = x_normal["Q_lag_6h"] = 0.0
xn_df = pd.DataFrame([x_normal])[app.FEATURE_COLS]
proba_n = float(modelo.predict_proba(xn_df)[0, 1])
estado_n = "ALERTA_6H" if proba_n > app.UMBRAL_ALERTA else "NORMAL"
print(f"  Escenario normal: proba={proba_n:.4f}  estado={estado_n}")
print("  predict_proba sin errores: OK")

print("\n=== VISTA OPERADOR tiene _simulador_modelo ===")
src_op = inspect.getsource(app.vista_operador)
assert "_simulador_modelo" in src_op, "vista_operador debe llamar _simulador_modelo"
print("  _simulador_modelo en vista_operador: OK")

print("\n=== TODOS LOS CHECKS PASAN ===")
