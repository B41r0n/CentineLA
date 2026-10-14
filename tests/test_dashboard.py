# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""Tests del dashboard: calendario, simulador y criterio de umbrales."""

import importlib.util
from itertools import pairwise
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("app", BASE_DIR / "dashboard" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_umbrales_inclusivos(app):
    # El semáforo, el gateway y 10_metricas_ternario.py usan proba >= umbral.
    assert app.clasificar_estado(0.2999) == "NORMAL"
    assert app.clasificar_estado(0.30) == "PRECAUCIÓN"
    assert app.clasificar_estado(0.6999) == "PRECAUCIÓN"
    assert app.clasificar_estado(0.70) == "ALERTA"


def test_calendario_dias_sin_lecturas_en_gris(app):
    ts = pd.to_datetime(["2026-10-01 10:00", "2026-10-03 10:00", "2026-10-03 11:00"])
    log = pd.DataFrame({"timestamp": ts, "estado": ["NORMAL", "ALERTA", "NORMAL"]})
    html = app._calendario_html(log, n_dias=3)
    assert '2026-10-01 · Sin alerta"' in html
    assert '2026-10-02 · Sin datos"' in html
    assert '2026-10-03 · Alerta roja"' in html


def test_simulador_acumulados_coherentes(app):
    valores = {"sim_p1": 2, "sim_r3": 6, "sim_r6": 12, "sim_r12": 18, "sim_r24": 30, "sim_r48": 54}
    x, ajustado = app.features_simulador(valores, pd.Timestamp("2026-10-01 14:00"))
    assert not ajustado
    assert list(x) == app.FEATURE_COLS
    assert x["roll_sum_24h"] == 30 and x["P_basin"] == 2
    assert x["lag_1h"] == pytest.approx((6 - 2) / 2)
    assert x["lag_24h"] == pytest.approx((54 - 30) / 24)
    assert x["Q_actual"] == pytest.approx(app._q_scs(30))
    assert x["hora_dia"] == 14 and x["mes"] == 10


def test_simulador_ajusta_acumulados_incoherentes(app):
    valores = {"sim_p1": 10, "sim_r3": 5, "sim_r6": 5, "sim_r12": 5, "sim_r24": 5, "sim_r48": 5}
    x, ajustado = app.features_simulador(valores, pd.Timestamp("2026-10-01 14:00"))
    assert ajustado
    assert x["roll_sum_3h"] == 10 and x["roll_sum_48h"] == 10


def test_q_scs_coincide_con_abstraccion_inicial(app):
    ia = 0.2 * ((25400 / app.CN) - 254)
    assert app._q_scs(ia) == 0.0
    assert app._q_scs(ia + 10) > 0


def test_marcador_del_medidor_tiene_estilo(app):
    # El marcador del gauge debe usar la clase que define el CSS (antes era invisible).
    assert 'class="mk"' in app._gauge_html(0.5)
    assert ".cl-gauge .mk{" in app.CSS_GLOBAL


def test_fecha_sin_lecturas_no_oculta_el_simulador():
    import datetime as dt

    from streamlit.testing.v1 import AppTest

    log = pd.read_csv(BASE_DIR / "data" / "processed" / "log_gateway_simulado.csv", parse_dates=["timestamp"])
    con_datos = set(log["timestamp"].dt.date)
    dias = pd.date_range(min(con_datos), max(con_datos), freq="D").date
    sin_datos = [d for d in dias if d not in con_datos]
    if not sin_datos:
        pytest.skip("el log no tiene días sin lecturas")

    at = AppTest.from_file(str(BASE_DIR / "dashboard" / "app.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Vista Operador (JAC)").run()
    fecha = next(d for d in at.date_input if d.label == "Seleccionar fecha")
    fecha.set_value(dt.date.fromisoformat(str(sin_datos[-1]))).run()
    assert not at.exception
    assert any("Sin lecturas" in w.value for w in at.warning)
    assert any(b.label == "Simular" for b in at.button)


def test_etiquetas_de_estado(app):
    etiquetas = {e: ui["etiqueta"] for e, ui in app.ESTADO_UI.items()}
    assert etiquetas == {"NORMAL": "Sin alerta", "PRECAUCIÓN": "Precaución", "ALERTA": "Alerta roja"}
    assert "Alerta roja" in app._pill("ALERTA", "Alerta roja")
    assert app.ESTADO_UI["ALERTA"]["pill_txt"] in app._pill("ALERTA", "Alerta roja")


def test_consulta_muestra_alerta_roja_con_su_color():
    import datetime as dt

    from streamlit.testing.v1 import AppTest

    log = pd.read_csv(BASE_DIR / "data" / "processed" / "log_gateway_simulado.csv", parse_dates=["timestamp"])
    fila = log[log["estado"] == "ALERTA"].iloc[-1]

    at = AppTest.from_file(str(BASE_DIR / "dashboard" / "app.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Vista Operador (JAC)").run()
    next(d for d in at.date_input if d.label == "Seleccionar fecha").set_value(
        dt.date.fromisoformat(str(fila["timestamp"].date()))
    ).run()
    next(s for s in at.selectbox if s.label == "Hora (hora local)").set_value(fila["timestamp"].strftime("%H:%M")).run()
    assert not at.exception
    estado = next(m.value for m in at.markdown if m.value.startswith("**Estado de alerta:**"))
    assert "Alerta roja" in estado and "#ff6b5e" in estado


def test_mapa_muestra_el_piloto_de_la_cedula(app):
    puntos = app.puntos_piloto()
    tipos = [p["tipo"] for p in puntos]
    assert tipos == ["pluviometro", "nodo", "nodo_camara", "nodo", "gateway"]
    nodos = [p for p in puntos if p["tipo"].startswith("nodo")]
    # Nodos cada ~500 m, el del medio en el punto del tramo medio
    medio = app.RECORRIDO[1]
    assert (nodos[1]["lat"], nodos[1]["lon"]) == (medio["lat"], medio["lon"])
    for a, b in pairwise(nodos):
        d_m = 1000 * app._haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        assert 490 <= d_m <= 510
    # Gateway junto al nodo del medio; pluviómetro en la cuenca alta
    gw = puntos[-1]
    assert 1000 * app._haversine_km(gw["lat"], gw["lon"], medio["lat"], medio["lon"]) < 300
    assert (puntos[0]["lat"], puntos[0]["lon"]) == (app.RECORRIDO[0]["lat"], app.RECORRIDO[0]["lon"])


def test_mapa_estatico_y_leyenda(app):
    # El respaldo sin internet se dibuja sin errores y la leyenda nombra cada equipo
    app._mapa_estatico(app.RECORRIDO, app.puntos_piloto(), "ALERTA")
    html = app._leyenda_mapa_html("ALERTA")
    for texto in ("Pluviómetro", "Nodo de cauce", "Nodo con cámara", "Gateway"):
        assert texto in html
