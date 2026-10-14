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

proxy = pd.read_csv(BASE_DIR / "data" / "processed" / "proxy_q_la_honda.csv", parse_dates=["timestamp"])
proxy = proxy.sort_values("timestamp").reset_index(drop=True)
log = pd.read_csv(BASE_DIR / "data" / "processed" / "log_gateway_simulado.csv", parse_dates=["timestamp"])
log["estado"] = log["proba_alerta"].apply(app.clasificar_estado)
import joblib
modelo = joblib.load(BASE_DIR / "simulate" / "models" / "clf_6h.joblib")

print("=== FIX 1 - Lluvia acumulada ===")
ult_24h, ult_7d, ult_30d, ref_fecha = app._lluvia_acumulada(proxy)
print(f"  ref_fecha (de proxy.max, NO del reloj): {ref_fecha}")
print(f"  ult_24h = {ult_24h:.3f}")
print(f"  ult_7d  = {ult_7d:.3f}")
print(f"  ult_30d = {ult_30d:.3f}")
assert isinstance(ref_fecha, str) and len(ref_fecha) == 10 and ref_fecha[4] == "-" and ref_fecha[7] == "-", \
    f"ref_fecha debe ser YYYY-MM-DD, se obtuvo: {ref_fecha}"
assert ref_fecha == str(proxy["timestamp"].max().date()), \
    "ref_fecha debe coincidir con proxy.max(), no con una fecha fija"
assert 0 <= ult_24h <= ult_7d <= ult_30d, "las ventanas deben ser crecientes (24 h <= 7 d <= 30 d)"
print("  OK: ventanas deslizantes crecientes y ref_fecha es proxy.max()")

print("\n=== FIX 2 - Mapa ===")
import inspect
src = inspect.getsource(app._mapa_folium)
assert "fit_bounds" in src, "debe usar fit_bounds"
assert "pad = 0.015" in src, "debe tener padding"
print("  fit_bounds presente en _mapa_folium: OK")
print("  padding definido: OK")

print("\n=== FIX 3 - Calendario: días sin lecturas en gris ===")
html_cal = app._calendario_html(log, n_dias=90)
ultima = log["timestamp"].dt.date.max()
dias = pd.date_range(ultima - pd.Timedelta(days=89), ultima, freq="D").date
sin_datos = sum(d not in set(log["timestamp"].dt.date) for d in dias)
celdas_grises = html_cal.count("sin datos\"")
print(f"  días sin lecturas en la ventana: {sin_datos} | celdas grises: {celdas_grises}")
assert celdas_grises == sin_datos, "cada día sin lecturas debe pintarse gris, no verde"
assert "Sin datos" in html_cal, "la leyenda debe incluir 'Sin datos'"
print("  OK")

print("\n=== FEATURE simulador ===")
assert hasattr(app, "_simulador_modelo"), "_simulador_modelo debe existir"
assert hasattr(app, "_ESCENARIOS"), "_ESCENARIOS debe existir"
assert len(app._ESCENARIOS) == 3, "deben ser 3 escenarios"
print("  _simulador_modelo y _ESCENARIOS presentes: OK")

ts_ref = pd.Timestamp(2026, 7, 1, 14)  # hora_dia=14, mes=7
resultados = {}
for nombre, valores in app._ESCENARIOS.items():
    x, ajustado = app.features_simulador(valores, ts_ref)
    assert not ajustado, f"el preset {nombre!r} debe ser coherente (acumulados crecientes)"
    x_df = pd.DataFrame([x])[app.FEATURE_COLS]
    proba = float(modelo.predict_proba(x_df)[0, 1])
    assert 0 <= proba <= 1, "proba debe estar en [0,1]"
    resultados[nombre] = (proba, app.clasificar_estado(proba))
    print(f"  {nombre}: proba={proba:.4f}  estado={resultados[nombre][1]}")

fuerte, aguacero, normal = (resultados[n][1] for n in app._ESCENARIOS)
assert fuerte == "ALERTA", "Lluvia fuerte sostenida debe dar ALERTA"
assert normal == "NORMAL", "Normal debe dar NORMAL con hora_dia=14, mes=7"
print("  predict_proba sin errores: OK")

x_inc, ajustado = app.features_simulador(
    {"sim_p1": 5, "sim_r3": 5, "sim_r6": 30, "sim_r12": 10, "sim_r24": 10, "sim_r48": 10}, ts_ref
)
assert ajustado and x_inc["roll_sum_12h"] == 30 and x_inc["roll_sum_48h"] == 30, \
    "acumulados incoherentes deben ajustarse a la ventana más corta"
print("  ajuste de acumulados incoherentes: OK")

print("\n=== VISTA OPERADOR tiene _simulador_modelo ===")
src_op = inspect.getsource(app.vista_operador)
assert "_simulador_modelo" in src_op, "vista_operador debe llamar _simulador_modelo"
print("  _simulador_modelo en vista_operador: OK")

print("\n=== TODOS LOS CHECKS PASAN ===")
