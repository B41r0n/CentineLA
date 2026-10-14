# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Bairon Nicolás Calle Rivera
"""
CentineLA — Dashboard Streamlit
Vista Pública + Vista Operador (JAC)

Uso:
    streamlit run dashboard/app.py
"""

import base64
import importlib.util
import io
import math
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── rutas base (relativas al repo, no al cwd de ejecución) ──────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "processed" / "log_gateway_simulado.csv"
PROXY_PATH = BASE_DIR / "data" / "processed" / "proxy_q_la_honda.csv"
MODEL_PATH = BASE_DIR / "simulate" / "models" / "clf_6h.joblib"

# Cooldown para el botón "Actualizar ahora" (segundos)
COOLDOWN_OK_S = 600
COOLDOWN_ERR_S = 60

# Métricas publicadas en "Confiabilidad del sistema": sistema (modelo + regla de lluvia
# seca), criterio proba >= umbral, test 2025-07-06 → 2026-09-30.
# Fuente: simulate/10_metricas_ternario.py (no se recalculan en vivo).
RECALL_PRECAUCION = 0.717026
PRECISION_PRECAUCION = 0.576108
RECALL_ALERTA = 0.551559
PRECISION_ALERTA = 0.867925

_MESES_ES = {
    "Jan": "ene", "Feb": "feb", "Mar": "mar", "Apr": "abr", "May": "may", "Jun": "jun",
    "Jul": "jul", "Aug": "ago", "Sep": "sep", "Oct": "oct", "Nov": "nov", "Dec": "dic",
}


@st.cache_resource
def _estado_actualizacion():
    """Estado compartido para evitar dobles clics y cooldown entre reruns."""
    return {"lock": threading.Lock(), "proximo": 0.0}


# Cargar simulate/11_actualizar_ahora.py via importlib (nombre con número)
_ACTUALIZAR_PATH = BASE_DIR / "simulate" / "11_actualizar_ahora.py"
_act_spec = importlib.util.spec_from_file_location("actualizar_ahora", _ACTUALIZAR_PATH)
_act_mod = importlib.util.module_from_spec(_act_spec)
_act_spec.loader.exec_module(_act_mod)
actualizar_ahora = _act_mod.actualizar_ahora
asegurar_db_sembrada = _act_mod._asegurar_db_sembrada

# CN compuesto de la cuenca (misma fuente que el proxy SCS-CN de 04_scs_cn_proxy.py)
_PARAMS_PATH = BASE_DIR / "simulate" / "cuenca_la_honda_params.py"
_params_spec = importlib.util.spec_from_file_location("cuenca_la_honda_params", _PARAMS_PATH)
_params_mod = importlib.util.module_from_spec(_params_spec)
_params_spec.loader.exec_module(_params_mod)
CN = _params_mod.CN

# Umbrales ternarios y regla de lluvia seca: única fuente compartida con 09 y 10
_REGLAS_PATH = BASE_DIR / "simulate" / "reglas_alerta.py"
_reglas_spec = importlib.util.spec_from_file_location("reglas_alerta", _REGLAS_PATH)
_reglas = importlib.util.module_from_spec(_reglas_spec)
_reglas_spec.loader.exec_module(_reglas)
UMBRAL_PRECAUCION = _reglas.UMBRAL_PRECAUCION
UMBRAL_ALERTA = _reglas.UMBRAL_ALERTA
PROBA_MAX_SIN_LLUVIA = _reglas.PROBA_MAX_SIN_LLUVIA
clasificar_estado = _reglas.clasificar_estado
ajustar_proba = _reglas.ajustar_proba

ZONA_HORARIA = "America/Bogota"

# ── tokens de diseño por estado (única fuente) ──
# Código de colores del semáforo: verde / amarillo / rojo. "etiqueta" es lo que ve el
# usuario; las claves (NORMAL / PRECAUCIÓN / ALERTA) son los nombres internos del log.
ESTADO_UI = {
    "NORMAL": {
        "color": "#2ecc71",
        "pill_txt": "#34d27b",
        "soft": "rgba(46,204,113,.14)",
        "zona": "rgba(46,204,113,.35)",
        "calendario": "#1f6f4a",  # verde apagado para que resalten los días con aviso
        "banner": "SIN ALERTA",
        "etiqueta": "Sin alerta",
        "accion": "No hay señales de alerta. Monitoreo activo.",
    },
    "PRECAUCIÓN": {
        "color": "#f1c40f",
        "pill_txt": "#f4d03f",
        "soft": "rgba(241,196,15,.14)",
        "zona": "rgba(241,196,15,.35)",
        "calendario": "#f1c40f",
        "banner": "PRECAUCIÓN — RIESGO MODERADO",
        "etiqueta": "Precaución",
        "accion": "La lluvia acumulada podría generar riesgo en las próximas 6 horas. Mantenerse atentos.",
    },
    "ALERTA": {
        "color": "#e74c3c",
        "pill_txt": "#ff6b5e",
        "soft": "rgba(231,76,60,.16)",
        "zona": "rgba(231,76,60,.35)",
        "calendario": "#e74c3c",
        "banner": "ALERTA ROJA — RIESGO EN LAS PRÓXIMAS 6 HORAS",
        "etiqueta": "Alerta roja",
        "accion": "Riesgo de creciente en las próximas 6 horas. Activar protocolo de la JAC.",
    },
}


# ── anclas geográficas de la quebrada (referencia visual aproximada, no GPS de campo) ─
ANCLAS = [
    {"nombre": "Cuenca Alta (Parque Arví)", "lat": 6.2804, "lon": -75.5027},
    {"nombre": "Tramo Medio (La Honda)", "lat": 6.2694, "lon": -75.5648},
    {"nombre": "Tramo Bajo (Moravia / Montecarlo)", "lat": 6.2781, "lon": -75.5670},
]
N_NODOS = 5

FEATURE_COLS = [
    "P_basin", "Q_actual",
    "lag_1h", "lag_3h", "lag_6h", "lag_12h", "lag_24h",
    "Q_lag_1h", "Q_lag_3h", "Q_lag_6h",
    "roll_sum_3h", "roll_sum_6h", "roll_sum_12h", "roll_sum_24h", "roll_sum_48h",
    "roll_max_6h", "hora_dia", "mes",
]


# ── nombres legibles para la UI (solo presentación, no cambian columnas internas) ──
NOMBRES_LEGIBLES = {
    "timestamp": "Fecha y hora",
    "P_basin": "Lluvia en la cuenca (mm/h)",
    "Q_actual": "Escorrentía actual",
    "Q_lag_1h": "Escorrentía hace 1 h",
    "Q_lag_3h": "Escorrentía hace 3 h",
    "Q_lag_6h": "Escorrentía hace 6 h",
    "lag_1h": "Lluvia hace 1 h",
    "lag_3h": "Lluvia hace 3 h",
    "lag_6h": "Lluvia hace 6 h",
    "lag_12h": "Lluvia hace 12 h",
    "lag_24h": "Lluvia hace 24 h",
    "roll_sum_3h": "Lluvia acumulada 3 h",
    "roll_sum_6h": "Lluvia acumulada 6 h",
    "roll_sum_12h": "Lluvia acumulada 12 h",
    "roll_sum_24h": "Lluvia acumulada 24 h",
    "roll_sum_48h": "Lluvia acumulada 48 h",
    "roll_max_6h": "Pico de lluvia en 6 h",
    "hora_dia": "Hora del día",
    "mes": "Mes",
    "proba_modelo": "Probabilidad del modelo (antes de la regla)",
    "proba_alerta": "Probabilidad de riesgo",
    "estado": "Estado de alerta",
}


# ── utilidades ────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def interpolar_nodos(anclas, n_total: int) -> list[dict]:
    """Genera n_total puntos a lo largo del recorrido real (alto→medio→bajo).

    Camina por longitud de arco (haversine) sobre los segmentos consecutivos
    Arví→La Honda y La Honda→Moravia, interpolando linealmente
    DENTRO de cada segmento por separado. Los nodos quedan repartidos
    proporcionalmente a la distancia real de cada tramo (no por parámetro t
    uniforme entre anclas, que ignora que un segmento puede ser mucho más
    largo que el otro y produce nodos mal distribuidos / recorrido torcido).
    """
    dist_acum = [0.0]
    for i in range(len(anclas) - 1):
        d = _haversine_km(
            anclas[i]["lat"], anclas[i]["lon"],
            anclas[i + 1]["lat"], anclas[i + 1]["lon"],
        )
        dist_acum.append(dist_acum[-1] + d)

    dist_total = dist_acum[-1]
    posiciones = np.linspace(0, dist_total, n_total)

    nodos = []
    for i, pos in enumerate(posiciones):
        seg_idx = len(dist_acum) - 2
        for s in range(len(dist_acum) - 1):
            if pos <= dist_acum[s + 1]:
                seg_idx = s
                break

        d_ini, d_fin = dist_acum[seg_idx], dist_acum[seg_idx + 1]
        frac = 0.0 if d_fin == d_ini else (pos - d_ini) / (d_fin - d_ini)

        ancla_ini, ancla_fin = anclas[seg_idx], anclas[seg_idx + 1]
        lat = ancla_ini["lat"] + frac * (ancla_fin["lat"] - ancla_ini["lat"])
        lon = ancla_ini["lon"] + frac * (ancla_fin["lon"] - ancla_ini["lon"])

        nodos.append({"nombre": f"Nodo {i + 1}", "lat": float(lat), "lon": float(lon)})

    return nodos


@st.cache_data(ttl=300, show_spinner=False)
def check_red(timeout: int = 3) -> bool:
    """Verifica conectividad a los tiles de OSM con timeout corto (cacheado 5 min)."""
    try:
        import requests
        requests.get("https://tile.openstreetmap.org/0/0/0.png", timeout=timeout)
        return True
    except Exception:
        return False


def _fecha_es(ts, formato: str) -> str:
    """strftime con abreviaturas de mes en español (independiente del locale)."""
    texto = ts.strftime(formato)
    for en, es in _MESES_ES.items():
        texto = texto.replace(en, es)
    return texto


def _compacto(html: str) -> str:
    """Elimina las líneas en blanco de un bloque HTML."""
    return "\n".join(linea for linea in html.split("\n") if linea.strip())


# ── carga de datos con caché ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_log() -> pd.DataFrame:
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Regla de lluvia seca (idempotente: el gateway ya la aplica) y estado ternario
    # recalculado, por si el log viene de una versión anterior del gateway.
    df["proba_alerta"] = ajustar_proba(df["proba_alerta"], df["roll_sum_24h"])
    df["estado"] = df["proba_alerta"].apply(clasificar_estado)
    return df


@st.cache_data(show_spinner=False)
def cargar_proxy() -> pd.DataFrame:
    df = pd.read_csv(PROXY_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return joblib.load(MODEL_PATH)


# ── componentes de mapa ───────────────────────────────────────────────────────

def _mapa_folium(anclas: list[dict], nodos: list[dict], estado: str) -> None:
    import folium
    from streamlit_folium import st_folium

    lats_a = [a["lat"] for a in anclas]
    lons_a = [a["lon"] for a in anclas]
    centro_lat = float(np.mean(lats_a))
    centro_lon = float(np.mean(lons_a))
    # OpenStreetMap con filtro CSS para tema oscuro (CartoDB exige API key)
    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    )
    # Filtro CSS solo sobre los tiles para oscurecer el mapa
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) "
        "brightness(.92) contrast(.88) saturate(.55);}</style>"))
    pad = 0.015   # ~1.5 km de margen alrededor de las anclas
    m.fit_bounds(
        [[min(lats_a) - pad, min(lons_a) - pad],
         [max(lats_a) + pad, max(lons_a) + pad]]
    )

    coords = [[a["lat"], a["lon"]] for a in anclas]
    folium.PolyLine(coords, color="#3498db", weight=3, opacity=0.7).add_to(m)

    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    color_nodo = ui["color"]
    for n in nodos:
        folium.CircleMarker(
            location=[n["lat"], n["lon"]],
            radius=10,
            color=color_nodo,
            fill=True,
            fill_color=color_nodo,
            fill_opacity=0.85,
            popup=folium.Popup(n["nombre"], parse_html=True),
            tooltip=n["nombre"],
        ).add_to(m)

    st_folium(m, height=550, use_container_width=True, returned_objects=[])
    st.caption("Mapa interactivo (OpenStreetMap, tema oscuro).")


def _mapa_estatico(anclas: list[dict], nodos: list[dict], estado: str) -> None:
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    lats_linea = [a["lat"] for a in anclas]
    lons_linea = [a["lon"] for a in anclas]
    lats_nodos = [n["lat"] for n in nodos]
    lons_nodos = [n["lon"] for n in nodos]

    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    color = ui["color"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#13171d")
    ax.set_facecolor("#13171d")

    # 1) Recorrido real — linea por las 3 anclas
    ax.plot(lons_linea, lats_linea, color="#3498db", linewidth=2.5, zorder=2)

    # 2) Marcadores de ancla
    lat_min = min(a["lat"] for a in anclas)
    lat_max_a = max(a["lat"] for a in anclas)
    lon_min_a = min(a["lon"] for a in anclas)
    lon_max_a = max(a["lon"] for a in anclas)
    lat_rng = (lat_max_a - lat_min) or 1.0
    lon_rng = (lon_max_a - lon_min_a) or 1.0

    for a in anclas:
        ax.scatter(
            [a["lon"]], [a["lat"]],
            s=200, facecolors="white", edgecolors="#3498db",
            linewidths=2.5, zorder=5, marker="o",
        )
        cerca_inferior = (a["lat"] - lat_min) / lat_rng < 0.25
        cerca_derecha = (lon_max_a - a["lon"]) / lon_rng < 0.25
        if cerca_inferior:
            txt_xy, txt_ha = (9, 10), "left"
        elif cerca_derecha:
            txt_xy, txt_ha = (-9, 10), "right"
        else:
            txt_xy, txt_ha = (9, -14), "left"
        ax.annotate(
            a["nombre"],
            (a["lon"], a["lat"]),
            textcoords="offset points",
            xytext=txt_xy,
            ha=txt_ha,
            fontsize=7, color="#8b95a5", style="italic",
        )

    # 3) Nodos de monitoreo
    ax.scatter(lons_nodos, lats_nodos, c=color, s=110, zorder=4, edgecolors="white", linewidths=1.5)

    ax.set_xlabel("Longitud", color="#e8ecf1")
    ax.set_ylabel("Latitud", color="#e8ecf1")
    ax.set_title(
        "CentineLA — Q. La Honda (posiciones ilustrativas, no GPS de campo)",
        fontsize=10, color="#e8ecf1",
    )
    ax.tick_params(colors="#8b95a5")
    ax.grid(alpha=0.1, color="#232830")

    # Leyenda
    handle_estado = mpatches.Patch(color=color, label=ui["etiqueta"])
    handle_ancla = mlines.Line2D(
        [], [], marker="o", color="w",
        markerfacecolor="white", markeredgecolor="#3498db",
        markersize=10, markeredgewidth=2.5, label="Ancla geográfica",
    )
    ax.legend(handles=[handle_estado, handle_ancla], loc="lower right", fontsize=8,
              facecolor="#13171d", edgecolor="#232830", labelcolor="#e8ecf1")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#13171d")
    plt.close(fig)
    buf.seek(0)
    st.image(
        buf.getvalue(),
        caption="⚠️ Mapa estático (sin conexión detectada). Posiciones ilustrativas, no GPS de campo.",
        width="stretch",
    )


def mostrar_mapa(anclas: list[dict], nodos: list[dict], estado: str) -> None:
    with st.spinner("Verificando conectividad para mapa interactivo..."):
        hay_red = check_red(timeout=3)

    if hay_red:
        try:
            _mapa_folium(anclas, nodos, estado)
        except Exception as exc:
            st.warning(f"Mapa interactivo no disponible ({exc}). Mostrando fallback estático.")
            _mapa_estatico(anclas, nodos, estado)
    else:
        st.info("Sin conexión a internet detectada. Mostrando mapa estático.")
        _mapa_estatico(anclas, nodos, estado)
    st.markdown(_leyenda_mapa_html(estado), unsafe_allow_html=True)


def _leyenda_mapa_html(estado: str) -> str:
    """Leyenda del mapa (sirve para el interactivo y el estático)."""
    color = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])["color"]

    def item(css: str, texto: str) -> str:
        return (f'<span><span style="display:inline-block;width:12px;height:12px;{css};'
                f'vertical-align:middle"></span> {texto}</span>')

    return (
        '<div class="cl-leyenda" style="display:flex;gap:16px;font-size:12px;'
        'color:var(--cl-muted);margin-top:6px">'
        + item(f"background:{color};border-radius:50%", "Nodo de monitoreo ilustrativo (color = estado actual)")
        + item("background:#3498db;height:3px;border-radius:2px", "Recorrido de la quebrada (alto → bajo)")
        + "</div>"
    )


# ── helpers de Vista Pública ──────────────────────────────────────────────────

def _color_proba(proba: float) -> str:
    return ESTADO_UI[clasificar_estado(proba)]["color"]


def _gauge_html(proba: float) -> str:
    """Gauge con tres zonas (verde/amarillo/rojo) y marcador."""
    p = max(0.0, min(1.0, proba))
    z_verde, z_amarilla, z_roja = (ESTADO_UI[e]["zona"] for e in ("NORMAL", "PRECAUCIÓN", "ALERTA"))
    w_green = UMBRAL_PRECAUCION * 100
    w_yellow = (UMBRAL_ALERTA - UMBRAL_PRECAUCION) * 100
    w_red = (1.0 - UMBRAL_ALERTA) * 100
    return f"""
<div class="cl-gauge">
  <div style="width:{w_green}%;background:{z_verde};border-radius:999px 0 0 999px;"></div>
  <div style="width:{w_yellow}%;background:{z_amarilla};"></div>
  <div style="width:{w_red}%;background:{z_roja};border-radius:0 999px 999px 0;"></div>
  <div class="mk" style="left:{p*100}%"></div>
</div>
<div class="cl-gauge-lbl">
  <span style="left:0%">0%</span>
  <span style="left:{UMBRAL_PRECAUCION:.0%}">{UMBRAL_PRECAUCION:.0%}</span>
  <span style="left:{UMBRAL_ALERTA:.0%}">{UMBRAL_ALERTA:.0%}</span>
  <span style="left:100%">100%</span>
</div>
"""


def _hero_html(estado: str, proba: float, ultima_ts_str: str) -> str:
    """Hero card: título (banner), acción, probabilidad grande, gauge, caption."""
    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    color_p = _color_proba(proba)
    return _compacto(f"""
<div class="cl-hero" style="--c:{ui['color']}">
  <div class="title">{ui['banner']}</div>
  <div class="action">{ui['accion']}</div>
  <div style="margin-top:14px">
    <div class="cl-stat" style="text-align:center">
      <div class="k">Probabilidad de riesgo (6 h)</div>
      <div class="v" style="color:{color_p}">{proba:.1%}</div>
    </div>
  </div>
  {_gauge_html(proba)}
  <div style="text-align:center;font-size:12px;color:var(--cl-muted);margin-top:8px">
    Semáforo calculado con la última lectura disponible ({ultima_ts_str}).
  </div>
  <div style="text-align:center;font-size:11px;color:var(--cl-muted);margin-top:4px;font-style:italic">
    Basado en lluvia regional (proxy); aún sin confirmación de sensores físicos en el cauce — hardware pendiente de instalación.
  </div>
</div>
""")


def _estado_semaforo_html(estado: str, proba: float) -> str:
    """Semaforo compacto para el simulador: pill + prob + gauge."""
    color_p = _color_proba(proba)
    return _compacto(f"""
<div class="cl-card" style="text-align:center;display:flex;flex-direction:column;align-items:center;gap:10px">
  <div>{_pill(estado)}</div>
  <div style="font-size:14px;color:var(--cl-muted)">Probabilidad de riesgo (6 h)</div>
  <div class="cl-stat" style="margin:4px 0"><div class="v" style="color:{color_p}">{proba:.1%}</div></div>
  {_gauge_html(proba)}
  <div style="font-size:11px;color:var(--cl-muted)">
    Verde < 30% · Amarillo 30–70% · Rojo ≥ 70%
  </div>
  <div style="text-align:center;font-size:11px;color:var(--cl-muted);margin-top:4px;font-style:italic">
    Basado en lluvia regional (proxy); aún sin confirmación de sensores físicos en el cauce — hardware pendiente de instalación.
  </div>
</div>
""")


def _tarjeta_html(icono: str, label: str, valor: str, unidad: str = "") -> str:
    return _compacto(f"""
<div class="cl-card cl-stat" style="text-align:center">
  <div style="font-size:1.6rem;margin-bottom:4px">{icono}</div>
  <div class="k">{label}</div>
  <div class="v">{valor}</div>
  <div class="u">{unidad}</div>
</div>
""")


def _calendario_html(log: pd.DataFrame, n_dias: int = 90) -> str:
    ultima_fecha = log["timestamp"].dt.date.max()
    inicio = ultima_fecha - pd.Timedelta(days=n_dias - 1)
    rango = pd.date_range(inicio, ultima_fecha, freq="D")

    # Estado máximo por día: 0=NORMAL, 1=PRECAUCIÓN, 2=ALERTA.
    # -1 = sin lecturas ese día (p. ej. menos de 2 estaciones): no es "sin riesgo".
    estado_map = {"NORMAL": 0, "PRECAUCIÓN": 1, "ALERTA": 2}
    estados = {0: "NORMAL", 1: "PRECAUCIÓN", 2: "ALERTA"}
    colores = {-1: "#3a3f47", **{n: ESTADO_UI[e]["calendario"] for n, e in estados.items()}}
    etiquetas = {-1: "Sin datos", **{n: ESTADO_UI[e]["etiqueta"] for n, e in estados.items()}}
    dia_estado = (
        log.groupby(log["timestamp"].dt.date)["estado"]
        .apply(lambda s: max((estado_map.get(e, 0) for e in s), default=0))
    )

    celdas = []
    for dia in rango:
        d = dia.date()
        nivel = int(dia_estado.get(d, -1))
        color = colores[nivel]
        celda = (
            '<div title="' + str(d) + " · " + etiquetas[nivel] + '" style="background:' + color
            + ';border-radius:4px;aspect-ratio:1;min-height:18px"></div>'
        )
        celdas.append(celda)

    grid_html = (
        '<div class="cl-cal" style="display:grid;grid-template-columns:repeat(18,1fr);'
        'gap:4px;max-width:100%;margin-bottom:8px">'
        + "".join(celdas)
        + "</div>"
    )
    leyenda = (
        '<div class="cl-leyenda" style="display:flex;gap:16px;font-size:12px;color:var(--cl-muted)">'
        + "&nbsp;&nbsp;".join(
            '<span><span style="display:inline-block;width:12px;height:12px;'
            f'background:{colores[n]};border-radius:2px;vertical-align:middle"></span> {etiquetas[n]}</span>'
            for n in (0, 1, 2, -1)
        )
        + "</div>"
    )
    return grid_html + leyenda


def _chips_html(items: list[tuple[str, str]], estado: str | None = None) -> str:
    """Fila de chips: items = [(label, valor_html), ...].
    Si estado no es None, el chip con label=='Estado' usa el color de ESTADO_UI[estado]."""
    html = ['<div class="cl-chips">']
    for label, valor in items:
        if estado and label == "Estado":
            ui = ESTADO_UI[estado]
            style = f'background:{ui["soft"]};border-color:{ui["color"]}'
        else:
            style = ""
        extra = f' style="{style}"' if style else ""
        html.append(f'<div class="cl-chip"{extra}><div class="k">{label}</div><div class="v">{valor}</div></div>')
    html.append('</div>')
    return "".join(html)


def _lluvia_acumulada(proxy: pd.DataFrame) -> tuple[float, float, float, str]:
    """Calcula lluvia acumulada en ventanas deslizantes desde la última fecha disponible.

    Usa proxy['timestamp'].max() como referencia — NUNCA la fecha real del sistema,
    para que las ventanas cuadren con la última lectura aunque los datos estén atrasados.
    Devuelve (ult_24h_mm, ult_7d_mm, ult_30d_mm, ref_fecha_str).

    P_basin es el promedio horario de las estaciones con dato, así que el acumulado
    también lo es. Se usan ventanas deslizantes en vez de día/mes del calendario
    para no mostrar 0 de forma confusa al inicio de cada mes.
    """
    ultima_ts = proxy["timestamp"].max()   # referencia fija al dataset, no al reloj

    p = proxy.dropna(subset=["P_basin"])

    def _acum(delta: pd.Timedelta) -> float:
        return float(p[p["timestamp"] >= ultima_ts - delta]["P_basin"].sum())

    ult_24h = _acum(pd.Timedelta(hours=24))
    ult_7d = _acum(pd.Timedelta(days=7))
    ult_30d = _acum(pd.Timedelta(days=30))
    ref_str = str(ultima_ts.date())
    return ult_24h, ult_7d, ult_30d, ref_str


# ── vista pública ─────────────────────────────────────────────────────────────

def vista_publica(log: pd.DataFrame, proxy: pd.DataFrame) -> None:
    ultima = log.iloc[-1]
    estado_actual = str(ultima["estado"])
    proba_actual = float(ultima["proba_alerta"])
    ultima_ts_str = _fecha_es(ultima["timestamp"], "%d-%b-%Y %H:%M")

    # 1. Fila de chips con datos reales (ancho completo)
    ult_24h, ult_7d, ult_30d, _ = _lluvia_acumulada(proxy)
    # Días con alerta en últimos 30 días
    corte_30d = ultima["timestamp"] - pd.Timedelta(days=30)
    log_30d = log[log["timestamp"] >= corte_30d]
    dias_alerta_30d = log_30d[log_30d["estado"] == "ALERTA"]["timestamp"].dt.date.nunique()

    chips = [
        ("Estado", _pill(estado_actual)),
        ("Probabilidad", f"{proba_actual:.1%}"),
        ("Lluvia 24 h", f"{ult_24h:.1f} mm"),
        ("Última lectura", ultima_ts_str),
        ("Días con alerta roja (30 d)", str(dias_alerta_30d)),
    ]
    st.markdown(_chips_html(chips, estado=estado_actual), unsafe_allow_html=True)

    # 2. Dos columnas: Hero (izq) + Últimos días con alertas (der)
    col_izq, col_der = st.columns([1, 1])

    with col_izq, st.container(border=True, key="card-hero"):
        st.markdown(_hero_html(estado_actual, proba_actual, ultima_ts_str), unsafe_allow_html=True)

    with col_der, st.container(border=True, key="card-alertas"):
        st.subheader("Últimos días con alertas")
        # Agrupar por fecha, tomar días con estado != NORMAL
        estado_map = {"NORMAL": 0, "PRECAUCIÓN": 1, "ALERTA": 2}
        dia_estado = (
            log.groupby(log["timestamp"].dt.date)["estado"]
            .apply(lambda s: max((estado_map.get(e, 0) for e in s), default=0))
        )
        dias_con_evento = [
            (d, e) for d, e in dia_estado.items() if e != 0
        ]
        dias_con_evento.sort(reverse=True)
        if dias_con_evento:
            items_html = []
            for d, nivel in dias_con_evento[:6]:
                est = {0: "NORMAL", 1: "PRECAUCIÓN", 2: "ALERTA"}[nivel]
                ui = ESTADO_UI[est]
                # probabilidad máxima del día
                mask_dia = log["timestamp"].dt.date == d
                proba_max = float(log[mask_dia]["proba_alerta"].max())
                fecha_str = _fecha_es(d, "%d-%b")
                items_html.append(f"""
<div class="cl-item">
  <span style="font-size:12px;color:var(--cl-muted)">{fecha_str}</span>
  <span>{_pill(est)}</span>
  <div class="cl-bar"><span style="width:{proba_max*100:.0f}%;background:{ui['color']}"></span></div>
  <span style="font-size:13px;font-weight:600;color:{ui['pill_txt']}">{proba_max:.0%}</span>
</div>""")
            st.markdown("".join(items_html), unsafe_allow_html=True)
        else:
            st.write("Sin precauciones ni alertas en el periodo.")
        st.caption(
            "Días recientes en que el sistema dio precaución o alerta. La "
            "barra es la probabilidad más alta de ese día."
        )

    # 3. Mapa a ancho completo
    with st.container(border=True, key="card-mapa"):
        st.subheader("Mapa de la quebrada y sus puntos de monitoreo")
        nodos = interpolar_nodos(ANCLAS, N_NODOS)
        mostrar_mapa(ANCLAS, nodos, estado_actual)
        st.caption(
            "Recorrido de la quebrada de la parte alta a la baja. Posiciones "
            "ilustrativas, no son GPS de campo. El piloto de la siguiente fase es "
            "1 pluviómetro en la cuenca alta, 3 nodos en el cauce del tramo medio "
            "(el del medio con cámara) y 1 gateway en una sede comunitaria; su "
            "ubicación exacta se define con visita de campo."
        )

    # 4. Lluvia acumulada: 3 tarjetas con _tarjeta_html en grid
    st.subheader("Lluvia acumulada")
    st.markdown(
        _compacto('<div class="cl-grid3">'
        + _tarjeta_html("🌧️", "Últimas 24 horas", f"{ult_24h:.1f}", "mm")
        + _tarjeta_html("📅", "Últimos 7 días", f"{ult_7d:.1f}", "mm")
        + _tarjeta_html("📆", "Últimos 30 días", f"{ult_30d:.1f}", "mm")
        + '</div>'),
        unsafe_allow_html=True,
    )
    st.caption(
        "Cuánta lluvia ha caído en la cuenca (promedio de las estaciones del IDEAM con "
        "dato). Más lluvia acumulada significa suelo más saturado y mayor riesgo."
    )

    # 5. Historial de los últimos 90 días
    st.subheader("Historial de los últimos 90 días")
    with st.container(border=True, key="card-calendario"):
        st.markdown(_calendario_html(log, n_dias=90), unsafe_allow_html=True)
    st.caption(
        "Cada cuadro es un día. Verde: sin alertas. Amarillo: hubo "
        "precaución. Rojo: hubo alerta roja. Gris: sin lecturas ese día (menos de 2 "
        "estaciones del IDEAM reportando), por lo que no se sabe el estado."
    )


def _estilo_plotly(fig: go.Figure) -> go.Figure:
    """Aplica tema oscuro a un figure de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8ecf1"),
        xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.06)"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


# ── helpers de Vista Operador ────────────────────────────────────────────────

def _grafico_historico(log: pd.DataFrame, proxy: pd.DataFrame) -> None:
    with st.container(border=True, key="card-historial"):
        st.subheader("📈 Comportamiento histórico de la cuenca")
        st.caption(
            "Azul: lluvia. Morado: agua que escurre hacia la quebrada. "
            "Puntos amarillos y rojos: momentos en que el sistema dio precaución o alerta."
        )

        fecha_min = proxy["timestamp"].dt.date.min()
        fecha_max = proxy["timestamp"].dt.date.max()
        fecha_inicio_default = max(
            fecha_min,
            (pd.Timestamp(fecha_max) - pd.Timedelta(days=90)).date(),
        )
        valor_inicial = (fecha_inicio_default, fecha_max)

        rango = st.date_input(
            "Rango de fechas",
            value=valor_inicial,
            min_value=fecha_min,
            max_value=fecha_max,
            key="rango_historico_op",
        )

        if not (isinstance(rango, (list, tuple)) and len(rango) == 2):
            st.info("Seleccionar fecha de inicio y fin del rango.")
            return

        fecha_ini, fecha_fin = rango
        mask_proxy = (
            (proxy["timestamp"].dt.date >= fecha_ini)
            & (proxy["timestamp"].dt.date <= fecha_fin)
        )
        proxy_rango = proxy[mask_proxy].copy()
        mask_log = (
            (log["timestamp"].dt.date >= fecha_ini)
            & (log["timestamp"].dt.date <= fecha_fin)
        )
        precauciones = log[mask_log & (log["estado"] == "PRECAUCIÓN")].copy()
        alertas = log[mask_log & (log["estado"] == "ALERTA")].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=proxy_rango["timestamp"], y=proxy_rango["P_basin"],
            name="Lluvia en la cuenca (mm/h)", line=dict(color="#3498db", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=proxy_rango["timestamp"], y=proxy_rango["Q_scs_proxy"],
            name="Escorrentía estimada (mm)", line=dict(color="#a569bd", width=1), yaxis="y2",
        ))
        if not precauciones.empty:
            fig.add_trace(go.Scatter(
                x=precauciones["timestamp"], y=precauciones["proba_alerta"],
                mode="markers", name=ESTADO_UI["PRECAUCIÓN"]["etiqueta"],
                marker=dict(color=ESTADO_UI["PRECAUCIÓN"]["color"], size=6, symbol="x"), yaxis="y2",
            ))
        if not alertas.empty:
            fig.add_trace(go.Scatter(
                x=alertas["timestamp"], y=alertas["proba_alerta"],
                mode="markers", name=ESTADO_UI["ALERTA"]["etiqueta"],
                marker=dict(color=ESTADO_UI["ALERTA"]["color"], size=6, symbol="x"), yaxis="y2",
            ))
            for ts in alertas["timestamp"].iloc[::max(1, len(alertas) // 200)]:
                fig.add_vline(x=ts, line=dict(color="rgba(231,76,60,.5)", width=0.6, dash="dot"), opacity=0.3)

        fig.update_layout(
            title=dict(
                text="Lluvia, escorrentía y alertas",
                y=0.98, yanchor="top", x=0, xanchor="left",
                font=dict(size=16),
            ),
            xaxis=dict(title="Fecha"),
            yaxis=dict(title=dict(text="Lluvia en la cuenca (mm/h)", font=dict(color="#3498db"))),
            yaxis2=dict(
                title=dict(text="Escorrentía estimada / probabilidad", font=dict(color="#a569bd")),
                overlaying="y", side="right",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.14, x=0),
            margin=dict(t=90, b=40, l=60, r=60),
            height=480,
        )
        _estilo_plotly(fig)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"En este período: {len(alertas)} horas en alerta roja y {len(precauciones)} en precaución. "
            "Las líneas verticales se muestran submuestreadas para rendimiento."
        )


# ── vista operador ────────────────────────────────────────────────────────────

def vista_operador(log: pd.DataFrame, proxy: pd.DataFrame, modelo) -> None:
    st.header("🛠️ Vista operador (JAC)")

    # 1. Registro de lecturas recientes
    with st.container(border=True, key="card-registro"):
        st.subheader("📋 Registro de lecturas recientes")
        st.caption(
            "Las últimas 100 lecturas horarias del sistema, la más reciente arriba. Cada "
            "fila muestra qué se midió y qué estado calculó."
        )
        log_mostrar = log.tail(100).iloc[::-1].copy()
        # Estado como lo lee la comunidad: Sin alerta / Precaución / Alerta roja
        log_mostrar["estado"] = log_mostrar["estado"].map(lambda e: ESTADO_UI[e]["etiqueta"])
        log_mostrar = log_mostrar.rename(columns=NOMBRES_LEGIBLES)
        numericas = log_mostrar.select_dtypes("number").columns
        log_mostrar[numericas] = log_mostrar[numericas].round(2)
        color_por_etiqueta = {ui["etiqueta"]: ui["pill_txt"] for ui in ESTADO_UI.values()}
        # Colorear columna Estado de alerta con pandas Styler
        try:
            def color_estado(val):
                color = color_por_etiqueta.get(val)
                return f"color: {color}; font-weight: 600" if color else ""
            styled = log_mostrar.style.map(color_estado, subset=["Estado de alerta"]).format(precision=2)
            st.dataframe(styled, width="stretch")
        except Exception:
            st.dataframe(log_mostrar, width="stretch")

    # 2. Comportamiento histórico de la cuenca
    _grafico_historico(log, proxy)

    # 3. Confiabilidad del sistema
    with st.container(border=True, key="card-confiabilidad"):
        st.subheader("🎯 Confiabilidad del sistema")
        st.caption(
            "Detectadas = de cada 100 horas de riesgo, cuántas avisó el sistema. "
            "Acertados = de cada 100 avisos, cuántos coincidieron con una hora de riesgo. "
            "Hora de riesgo = el proxy hidrológico (SCS-CN) estima escorrentía hacia la "
            "quebrada en las 6 h siguientes; no son crecientes medidas en campo. "
            "Calculado sobre el periodo de prueba (2025-07-06 → 2026-09-30, 9,056 horas), "
            "que el modelo no vio al entrenar. Incluye la regla de lluvia seca (sin lluvia "
            "en 24 h no hay aviso). No se recalcula en vivo — "
            "fuente: simulate/10_metricas_ternario.py."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Horas de riesgo detectadas (desde precaución)",
                f"{RECALL_PRECAUCION:.1%}",
                help="Recall con probabilidad ≥ 0.30 (Precaución o más). Fuente: simulate/10_metricas_ternario.py",
            )
            st.metric(
                "Horas de riesgo detectadas (solo alerta roja)",
                f"{RECALL_ALERTA:.1%}",
                help="Recall con probabilidad ≥ 0.70 (Alerta roja). Fuente: simulate/10_metricas_ternario.py",
            )
        with c2:
            st.metric(
                "Avisos acertados (desde precaución)",
                f"{PRECISION_PRECAUCION:.1%}",
                help="Precisión con probabilidad ≥ 0.30 (Precaución o más). Fuente: simulate/10_metricas_ternario.py",
            )
            st.metric(
                "Alertas rojas acertadas",
                f"{PRECISION_ALERTA:.1%}",
                help="Precisión con probabilidad ≥ 0.70 (Alerta roja). Fuente: simulate/10_metricas_ternario.py",
            )
        st.caption(
            "Sin la regla de lluvia seca (aplicada desde el 13-oct-2026), el mismo modelo "
            "acierta 49.2% de sus avisos desde precaución y 85.5% de sus alertas rojas, con "
            "las mismas horas detectadas: la regla solo quita avisos en horas sin lluvia."
        )

    # 4. Qué factores pesan más en la alerta
    with st.container(border=True, key="card-factores"):
        st.subheader("🔍 ¿Qué factores pesan más en la alerta?")
        st.caption(
            "La lluvia acumulada de las últimas 24–48 horas es la señal más importante. "
            "Un aguacero puntual sin antecedentes genera menos riesgo que lluvia "
            "sostenida por días."
        )
        try:
            # Nombres tomados del propio modelo para no depender del orden de FEATURE_COLS
            nombres = getattr(modelo, "feature_names_in_", FEATURE_COLS)
            importancias = dict(zip(nombres, modelo.feature_importances_, strict=True))
            df_imp = (
                pd.DataFrame(
                    {"feature": list(importancias.keys()), "importance": list(importancias.values())}
                )
                .sort_values("importance", ascending=True)
            )
            df_imp["feature_legible"] = df_imp["feature"].map(NOMBRES_LEGIBLES).fillna(df_imp["feature"])
            fig_imp = go.Figure(go.Bar(
                x=df_imp["importance"],
                y=df_imp["feature_legible"],
                orientation="h",
                marker_color="#3498db",
            ))
            fig_imp.update_layout(
                height=500,
                title="Peso de cada factor en la decisión del modelo",
                margin=dict(l=220),
                xaxis_title="Importancia relativa",
            )
            _estilo_plotly(fig_imp)
            st.plotly_chart(fig_imp, width="stretch")
            with st.expander("Detalle técnico"):
                st.write(
                    "Modelo: RandomForestClassifier(n_estimators=200, random_state=42, "
                    "class_weight='balanced') — archivo: simulate/models/clf_6h.joblib"
                )
                st.write("Features usadas:", ", ".join(FEATURE_COLS))
        except Exception as exc:
            st.warning(f"No se pudo graficar importancias: {exc}")

    # 5. Consultar un momento específico
    with st.container(border=True, key="card-consulta"):
        st.subheader("🔎 Consultar un momento específico")
        st.caption(
            "Elige una fecha y hora para ver qué datos recibió el sistema en ese momento "
            "y qué estado calculó."
        )
        fecha_min_log = log["timestamp"].dt.date.min()
        fecha_max_log = log["timestamp"].dt.date.max()

        fecha_sel = st.date_input(
            "Seleccionar fecha",
            value=fecha_max_log,
            min_value=fecha_min_log,
            max_value=fecha_max_log,
            key="fecha_inspeccion",
        )
        log_dia = log[log["timestamp"].dt.date == fecha_sel].copy()

        # Sin `return`: una fecha sin lecturas no debe ocultar el simulador de abajo.
        if log_dia.empty:
            st.warning("Sin lecturas para esta fecha (menos de 2 estaciones reportando). Elige otra.")
        else:
            _detalle_lectura(log_dia)

    # 6. Simulador
    with st.container(border=True, key="card-simulador"):
        _simulador_modelo(modelo)


def _detalle_lectura(log_dia: pd.DataFrame) -> None:
    """Selector de hora y detalle de una lectura del log (sección 'Consultar un momento')."""
    opciones_hora = log_dia["timestamp"].dt.strftime("%H:%M").tolist()
    hora_sel = st.selectbox("Hora (hora local)", opciones_hora, key="hora_inspeccion")

    fila = log_dia[log_dia["timestamp"].dt.strftime("%H:%M") == hora_sel].iloc[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Fecha y hora:** `{fila['timestamp']}`")
        st.markdown(f"**Estado de alerta:** {_pill(fila['estado'])}", unsafe_allow_html=True)
        st.markdown(f"**Probabilidad de riesgo:** `{fila['proba_alerta']:.4f}`")

    with col_b:
        cols_mostrar = [c for c in ["P_basin", "Q_actual", "Q_lag_1h", "roll_sum_24h", "hora_dia", "mes"]
                        if c in fila.index]
        df_vals = pd.DataFrame([{c: round(float(fila[c]), 6) for c in cols_mostrar}])
        df_vals = df_vals.T.rename(columns={0: "Valor"})
        df_vals.index = [NOMBRES_LEGIBLES.get(idx, idx) for idx in df_vals.index]
        st.dataframe(df_vals, width="stretch")


# ── simulador interactivo del modelo ─────────────────────────────────────────

# Controles del simulador: (clave, etiqueta, máximo en mm). Cada uno es lluvia
# promedio de la cuenca; los acumulados incluyen la última hora.
_SIM_CONTROLES = [
    ("sim_p1", "Lluvia en la última hora (mm)", 40.0),
    ("sim_r3", "Lluvia acumulada 3 h (mm)", 60.0),
    ("sim_r6", "Lluvia acumulada 6 h (mm)", 80.0),
    ("sim_r12", "Lluvia acumulada 12 h (mm)", 100.0),
    ("sim_r24", "Lluvia acumulada 24 h (mm)", 120.0),
    ("sim_r48", "Lluvia acumulada 48 h (mm)", 150.0),
]

_ESCENARIOS = {
    "🌧️ Lluvia fuerte sostenida": {
        "sim_p1": 6.0, "sim_r3": 15.0, "sim_r6": 25.0,
        "sim_r12": 35.0, "sim_r24": 45.0, "sim_r48": 55.0,
    },
    "⛈️ Aguacero puntual": {
        "sim_p1": 8.0, "sim_r3": 10.0, "sim_r6": 10.0,
        "sim_r12": 10.0, "sim_r24": 10.0, "sim_r48": 10.0,
    },
    "☀️ Normal": {
        "sim_p1": 0.0, "sim_r3": 0.0, "sim_r6": 0.0,
        "sim_r12": 0.0, "sim_r24": 0.0, "sim_r48": 0.0,
    },
}


def _q_scs(p_acc_24h: float) -> float:
    """Escorrentía SCS-CN (mm) para un acumulado de 24 h, igual que 04_scs_cn_proxy.py."""
    s = (25400 / CN) - 254
    ia = 0.2 * s
    if p_acc_24h <= ia:
        return 0.0
    exceso = p_acc_24h - ia
    return exceso**2 / (exceso + s)


def features_simulador(valores: dict, ahora: pd.Timestamp) -> tuple[dict, bool]:
    """Convierte los controles del simulador en las 18 features del modelo.

    Los acumulados se fuerzan a ser crecientes (1 h ≤ 3 h ≤ … ≤ 48 h); si hubo que
    ajustarlos se devuelve ajustado=True. Las lluvias hora a hora (lags) se
    aproximan repartiendo uniformemente la lluvia de cada tramo entre ventanas, y la
    escorrentía se calcula con SCS-CN sobre el acumulado de 24 h (actual y estimado
    hace 1, 3 y 6 h), como hace el proxy. Devuelve (features, ajustado).
    """
    crudos = [float(valores[k]) for k, _, _ in _SIM_CONTROLES]
    acum = list(np.maximum.accumulate(crudos))
    ajustado = acum != crudos
    p1, r3, r6, r12, r24, r48 = acum

    def _r24_hace(horas: int, lluvia_reciente: float) -> float:
        # Acumulado de 24 h que había hace `horas`: sale la lluvia reciente y
        # entra la de hace 24–48 h (repartida uniformemente).
        return max(0.0, r24 - lluvia_reciente + (r48 - r24) * horas / 24)

    lag_1h = (r3 - p1) / 2
    lag_3h = (r6 - r3) / 3
    x = {
        "P_basin": p1,
        "Q_actual": _q_scs(r24),
        "lag_1h": lag_1h,
        "lag_3h": lag_3h,
        "lag_6h": (r12 - r6) / 6,
        "lag_12h": (r24 - r12) / 12,
        "lag_24h": (r48 - r24) / 24,
        "Q_lag_1h": _q_scs(_r24_hace(1, p1)),
        "Q_lag_3h": _q_scs(_r24_hace(3, r3)),
        "Q_lag_6h": _q_scs(_r24_hace(6, r6)),
        "roll_sum_3h": r3,
        "roll_sum_6h": r6,
        "roll_sum_12h": r12,
        "roll_sum_24h": r24,
        "roll_sum_48h": r48,
        "roll_max_6h": max(p1, lag_1h, lag_3h),
        "hora_dia": float(ahora.hour),
        "mes": float(ahora.month),
    }
    return x, ajustado


# ── contenido de la guía del proyecto ─────────────────────────────────────────

_GUIA_STATS = [
    {"valor": "6 h", "etiqueta": "anticipación"},
    {"valor": "2", "etiqueta": "estaciones IDEAM activas"},
    {"valor": "44,372", "etiqueta": "horas útiles (2016 – 30-sep-2026)"},
    {"valor": "71.7%", "etiqueta": "detección en amarillo"},
    {"valor": "86.8%", "etiqueta": "precisión en rojo"},
    {"valor": "18", "etiqueta": "variables del modelo"},
]

_GUIA_PIPELINE = [
    {
        "num": 1,
        "titulo": "Lluvia",
        "desc": "Estaciones oficiales del IDEAM, datos abiertos",
        "estado": "Operando hoy",
        "icono": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M20 16.2A4.5 4.5 0 0 0 3.2 14.2C1.4 12.4 1 10 2.5 8s4-3 6-2a6 6 0 0 1 11.5 2.2"/></svg>""",
    },
    {
        "num": 2,
        "titulo": "Proxy hidrológico",
        "desc": "Estima cuánta agua escurre hacia la quebrada (método SCS-CN)",
        "estado": "Operando hoy",
        "icono": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.7a6 6 0 0 1 6 6c0 3.3-6 12.3-6 12.3S6 12 6 8.7a6 6 0 0 1 6-6Z"/></svg>""",
    },
    {
        "num": 3,
        "titulo": "Modelo predictivo",
        "desc": "Probabilidad de riesgo a 6 horas (Random Forest)",
        "estado": "Operando hoy",
        "icono": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16Z"/><path d="M12 6v6l4 2"/></svg>""",
    },
    {
        "num": 4,
        "titulo": "Semáforo y tablero",
        "desc": "Vista pública y vista operador para la JAC",
        "estado": "Operando hoy",
        "icono": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>""",
    },
    {
        "num": 5,
        "titulo": "Nodos en el cauce y alertas",
        "desc": "Sensores de nivel, cámara, semáforo, sirena, WhatsApp comunitario y DAGRD",
        "estado": "Siguiente fase",
        "icono": """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9a6 6 0 0 1 12 0c0 4-6 9-6 9S6 13 6 9Z"/><circle cx="12" cy="9" r="2"/></svg>""",
    },
]

_GUIA_VISTAS = [
    {
        "titulo": "Vista pública",
        "items": ["Semáforo de riesgo", "Probabilidad de riesgo a 6 horas", "Lluvia acumulada", "Mapa de la quebrada", "Calendario de alertas de 90 días"],
    },
    {
        "titulo": "Vista operador (JAC)",
        "items": ["Registro de lecturas", "Histórico de la cuenca", "Métricas de confiabilidad", "Importancia de variables", "Simulador de escenarios"],
    },
]

_GUIA_TECH_CHIPS = [
    "Python", "scikit-learn", "Streamlit", "Folium", "SQLite",
    "API Socrata (datos.gov.co)", "GitHub",
]

_GUIA_PROD_URL = "https://centinela-s4h6yg7ppapp2udzgrugdgt.streamlit.app/"

_GUIA_REPO = """CentineLA/
├─ dashboard/        app.py (tablero Streamlit) · _check_fixes.py
├─ simulate/         04 proxy SCS-CN · 05 validación · 06 variables
│                    07–08 modelo · 09 gateway simulado · 10 métricas
│                    11 actualizar ahora
├─ scripts/          01 estaciones · 02 descarga histórica del IDEAM
├─ data/processed/   datasets y proxy horario
├─ .streamlit/       tema oscuro
├─ tests/           pruebas (pytest)
└─ README.md · LICENSE (AGPL-3.0) · requirements.txt · requirements-dev.txt"""

_GUIA_FAQ = [
    {
        "cat": "El proyecto",
        "q": "¿Qué problema resuelve CentineLA?",
        "simple": "La Quebrada La Honda, en la Comuna 4 Aranjuez de Medellín, no tiene un sistema de alerta temprana instrumentado y ha tenido crecientes súbitas (1996, 2022 y abril de 2026). CentineLA busca avisar con horas de anticipación para que la comunidad pueda reaccionar.",
        "tecnico": "En 1996 el evento dejó cerca de 260 personas damnificadas y 17 heridos. El DAGRD ya opera alertas en otras quebradas de la misma comuna (La Rosa y La Bermejala), pero no en La Honda.",
    },
    {
        "cat": "El proyecto",
        "q": "¿Cómo funciona, en pocas palabras?",
        "simple": "Toma la lluvia que reportan estaciones oficiales del IDEAM, estima cuánta agua escurre hacia la quebrada y un modelo calcula la probabilidad de que eso ocurra en las próximas 6 horas. Se muestra como un semáforo.",
        "tecnico": "API Socrata del IDEAM (dataset s54a-sgyg) → promedio de lluvia de la cuenca → proxy SCS-CN → 18 variables (lluvia acumulada de 3 a 48 h, escorrentía, hora y mes) → RandomForestClassifier → bandas de probabilidad.",
    },
    {
        "cat": "El proyecto",
        "q": "¿Qué significan verde, amarillo y rojo?",
        "simple": "Sin alerta (verde): sin señales de riesgo. Precaución (amarillo): conviene vigilar. Alerta roja (rojo): probabilidad alta de lluvia suficiente para generar escorrentía importante en las próximas 6 horas.",
        "tecnico": "Bandas de probabilidad: &lt;0.30 Sin alerta · 0.30–0.70 Precaución · ≥0.70 Alerta roja (en el código: NORMAL / PRECAUCIÓN / ALERTA). Sin lluvia en 24 h no hay aviso.",
    },
    {
        "cat": "El proyecto",
        "q": "¿Quién lo desarrolló y es un sistema oficial?",
        "simple": "Es un prototipo académico de Bairon Nicolás Calle Rivera (Gestión de Redes de Telecomunicaciones, ITM) para Territorio INN 2026, Reto #7. No reemplaza los protocolos oficiales de gestión del riesgo: busca apoyarlos.",
        "tecnico": "Código abierto bajo licencia AGPL-3.0, disponible en GitHub.",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿De dónde salen los datos?",
        "simple": "De estaciones oficiales del IDEAM, publicadas como datos abiertos de Colombia.",
        "tecnico": "Dataset s54a-sgyg. Activas: Metromedellín (0027015310) y Olaya Herrera, sensor 240 (0027015330). Pajarito (0027015290) está suspendida desde 2020: su histórico 2016–2020 sí entra al promedio de lluvia y al entrenamiento, pero no a la actualización en vivo. 44,372 horas utilizables tras los filtros de calidad (corte de evaluación 30-sep-2026)."
    },
    {
        "cat": "Datos y modelo",
        "q": "¿La lluvia que se usa es la de la quebrada?",
        "simple": "No exactamente: no hay estaciones oficiales dentro de la microcuenca, así que se usan las más cercanas como aproximación.",
        "tecnico": "Están entre 5.7 y 6.7 km. Es un proxy regional, limitación documentada. Se solicitó información instrumental adicional al Área Metropolitana del Valle de Aburrá (SIATA).",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Qué es el método SCS-CN y de dónde sale el 81.1?",
        "simple": "Es un método estándar que estima cuánta lluvia se convierte en agua que escurre, según el suelo y el uso del terreno. El valor para La Honda es una estimación de ingeniería, no una medición oficial.",
        "tecnico": "Reparto de área 40/30/30 (tramo alto/medio/bajo) con CN típicos 70/92/85 y suelo tipo C asumido → CN compuesto 81.1. Variar los CN por tramo ±10% mueve el CN compuesto ±3.1–3.5%. Método: USDA NRCS (2004).",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Qué predice exactamente el modelo?",
        "simple": "La probabilidad de que dentro de 6 horas la cuenca esté en condición de escurrir agua de forma importante, es decir, que la lluvia acumulada de 24 horas supere lo que el terreno alcanza a absorber.",
        "tecnico": "Clase positiva = escorrentía SCS-CN a 6 h &gt; 0, equivalente a lluvia acumulada en 24 h por encima de ~11.8 mm (abstracción inicial con CN 81.1). Entrenamiento: 3,138 horas positivas de 35,316. Prueba: 417 de 9,056.",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Qué tan confiable es?",
        "simple": "En pruebas con datos que el modelo no vio al entrenar, el nivel amarillo detecta cerca de 7 de cada 10 situaciones de riesgo y, cuando se activa el rojo, acierta en más de 8 de cada 10 veces.",
        "tecnico": "Validación cronológica 80/20 (corte 2025-07-06 16:00; prueba hasta 2026-09-30). Umbral ≥0.30: recall 71.7% · precisión 57.6%. Umbral ≥0.70: recall 55.2% · precisión 86.8%. Incluyen la regla de lluvia seca (sin lluvia en 24 h no hay aviso). Las métricas son frente al proxy hidrológico. Sin la regla, el mismo modelo da 49.2% y 85.5% de precisión con el mismo recall.",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Por qué prefieren avisar de más que de menos?",
        "simple": "Una falsa alarma cuesta un aviso de más; una creciente sin aviso puede costar vidas. Por eso el amarillo se diseñó para detectar la mayor cantidad posible de situaciones de riesgo.",
        "tecnico": "El umbral bajo (0.30) prioriza recall; el alto (0.70) prioriza precisión para que el rojo sea más confiable.",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Por qué puede marcar alerta alta solo con lluvia?",
        "simple": "Porque hoy CentineLA mide riesgo por lluvia: con unos 22 mm acumulados en un día ya se supera casi al doble el umbral en que el terreno empieza a escurrir. Los sensores en el cauce, de la siguiente fase, servirán para confirmar el nivel real antes de escalar una alerta.",
        "tecnico": "El diseño completo exige al menos 2 señales críticas simultáneas (modelo + sensor físico de nivel o turbidez) para una alerta roja confirmada. Las variables más influyentes: lluvia acumulada 24 h (20.6%) y 48 h (12.8%). La probabilidad refleja la confianza dentro de la definición del modelo, no una medición del nivel del agua. Sin lluvia en las últimas 24 h el sistema no avisa: el modelo, apoyado en la hora y el mes, a veces daba precaución o alerta en horas secas (89 avisos en el periodo de prueba, ninguno acertado), así que en esas horas la probabilidad se limita a la frecuencia histórica (0.66%).",
    },
    {
        "cat": "Datos y modelo",
        "q": "¿Ya se validó contra crecientes reales?",
        "simple": "Aún no: es el siguiente paso. Las crecientes documentadas son contexto del proyecto; faltan registros oficiales de nivel para contrastar el modelo con lo que realmente ocurrió.",
        "tecnico": "Se solicitó al Área Metropolitana (SIATA) series de nivel y caudal y el registro de los eventos de 1996 y abril de 2026. Con esos datos se podría calibrar el proxy y ampliar la validación.",
    },
    {
        "cat": "Plataforma",
        "q": "¿Ya hay sensores instalados en la quebrada?",
        "simple": "Todavía no. Hoy funciona completa la parte de datos, predicción y tablero; los nodos físicos son la siguiente fase.",
        "tecnico": "Piloto (siguiente fase): 3 nodos ESP32 en el cauce del tramo medio, cada ~500 m, con sensor de nivel ultrasónico, turbidez, semáforo LED, sirena y panel solar (el del medio además con cámara IP), y un pluviómetro en la cuenca alta, conectados por LoRa a un gateway Raspberry Pi 4 en una sede comunitaria. Cubrir todo el cauce (6.2 km) requiere un nodo cada ~500 m. Un gateway simulado valida que la inferencia en streaming coincide con la de lote (tolerancia 1e-6, 500 marcas de tiempo aleatorias). Instalar requiere visita de campo y coordinación con la Secretaría de Medio Ambiente.",
    },
    {
        "cat": "Plataforma",
        "q": "¿Los puntos del mapa son donde irán los sensores?",
        "simple": "No, son ilustrativos: representan el recorrido de la quebrada por sus tramos alto, medio y bajo. El piloto de la cédula tiene 1 pluviómetro arriba, 3 nodos en el tramo medio y 1 gateway; la ubicación real se define con visita de campo.",
        "tecnico": "3 puntos de referencia y 5 nodos interpolados entre ellos, sin coordenadas GPS de campo. El piloto (siguiente fase) es 1 pluviómetro en la cuenca alta, 3 nodos de cauce cada ~500 m en el tramo medio (el del medio con cámara) y 1 gateway en una sede comunitaria.",
    },
    {
        "cat": "Plataforma",
        "q": "¿Por qué está en Streamlit?",
        "simple": "Para que cualquiera abra el tablero desde el navegador, sin instalar nada, y para poder mejorarlo rápido.",
        "tecnico": "Streamlit permite construir el tablero en Python, el mismo lenguaje del modelo. Se despliega en Streamlit Community Cloud conectado a GitHub: cada cambio al código se publica solo. Plan gratuito.",
    },
    {
        "cat": "Plataforma",
        "q": "¿Por qué a veces los datos vuelven a una fecha anterior?",
        "simple": "La versión gratuita se \"duerme\" cuando nadie la usa y al despertar carga los datos guardados en el repositorio. El botón Actualizar ahora trae los más recientes del IDEAM.",
        "tecnico": "El plan gratuito no tiene disco persistente: el tablero vuelve a leer los CSV del repositorio (la copia en SQLite, centinela.db, se regenera desde ellos y es solo un respaldo para consultas externas). Una base externa o una actualización programada (GitHub Actions) lo resolvería; queda como mejora.",
    },
    {
        "cat": "Alcance",
        "q": "¿Se puede replicar en otras quebradas?",
        "simple": "Sí. La arquitectura sirve para otras microcuencas con estaciones cercanas; cada una necesita sus propios parámetros.",
        "tecnico": "Cada cuenca requiere su CN, parámetros morfométricos y validación con eventos documentados.",
    },
    {
        "cat": "Alcance",
        "q": "¿Qué sigue?",
        "simple": "Instalar un nodo piloto, validar con datos reales de nivel y conectar los avisos a la comunidad y al DAGRD.",
        "tecnico": "Incorporar la respuesta del SIATA como validación o variable nueva; evaluar regresión sobre variable continua si crece el volumen de datos; canal de aviso por WhatsApp comunitario y DAGRD.",
    },
]


def _simulador_modelo(modelo) -> None:
    st.subheader("🧪 Simular un escenario de lluvia")
    st.caption(
        "Ingresa condiciones hipotéticas para ver cómo respondería el sistema. Útil "
        "para entender cuándo se activa cada estado."
    )

    # ── escenarios rápidos ────────────────────────────────────────────────
    cols_btn = st.columns(len(_ESCENARIOS))
    for col, (nombre, vals) in zip(cols_btn, _ESCENARIOS.items(), strict=True):
        if col.button(nombre, width="stretch", key="btn_" + nombre[:6]):
            for k, v in vals.items():
                st.session_state[k] = v

    # ── sliders ───────────────────────────────────────────────────────────
    valores = {}
    col1, col2 = st.columns(2)
    for i, (clave, etiqueta, maximo) in enumerate(_SIM_CONTROLES):
        with (col1 if i < 3 else col2):
            valores[clave] = st.slider(etiqueta, 0.0, maximo, step=0.5, key=clave)

    # ── features derivadas (aproximación) ────────────────────────────────
    ahora = pd.Timestamp.now(tz=ZONA_HORARIA)
    x_vec, ajustado = features_simulador(valores, ahora)

    if st.button("Simular", type="primary"):
        x_df = pd.DataFrame([x_vec])[FEATURE_COLS]
        proba_modelo = float(modelo.predict_proba(x_df)[0, 1])
        proba = ajustar_proba(proba_modelo, x_vec["roll_sum_24h"])
        estado_pred = clasificar_estado(proba)
        st.markdown(
            _estado_semaforo_html(estado_pred, proba),
            unsafe_allow_html=True,
        )
        if ajustado:
            st.warning(
                "Algunos acumulados eran menores que una ventana más corta (por ejemplo, "
                "menos lluvia en 24 h que en 6 h). Para simular se igualaron al valor "
                "de la ventana más corta."
            )
        if clasificar_estado(proba_modelo) != estado_pred:
            st.info(
                f"Sin lluvia en las últimas 24 h: el modelo daba {proba_modelo:.1%}, pero sin "
                "lluvia previa harían falta ~12 mm nuevos en 6 h para que haya riesgo, algo que "
                "el modelo no puede anticipar. Se usa la frecuencia histórica de esas horas "
                f"({PROBA_MAX_SIN_LLUVIA:.2%})."
            )
        st.caption(
            f"Probabilidad calculada: {proba:.1%}. "
            "Verde menos de 30% · Amarillo 30–70% · Rojo 70% o más. "
            f"La escorrentía se calcula con SCS-CN (CN {CN}) a partir del acumulado de 24 h; "
            "la lluvia hora a hora se aproxima repartiendo cada acumulado de forma pareja. "
            f"Hora y mes: los actuales en Colombia ({ahora:%H:%M}, mes {ahora.month})."
        )




def _guia_block(html: str) -> str:
    return "\n".join(line for line in textwrap.dedent(html).splitlines() if line.strip())


_GUIA_CSS = _guia_block("""
<style>
.cl-guia {
  color:#e8ecf1;
  background:
    radial-gradient(circle at 8% 12%, rgba(52,152,219,.12), transparent 30%),
    radial-gradient(circle at 92% 88%, rgba(231,76,60,.07), transparent 30%),
    #0b0d11;
  padding:2px;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
}
.cl-guia a {color:#3498db;text-decoration:none}
.cl-guia a:hover {text-decoration:underline}
.cl-guia-hero {display:flex;align-items:center;gap:16px;padding:22px;border-radius:16px;background:linear-gradient(145deg, rgba(52,152,219,.10), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);margin-bottom:20px}
.cl-guia-hero img {width:64px;height:80px;object-fit:contain;flex-shrink:0}
.cl-guia-hero-title {font-size:32px;font-weight:700;letter-spacing:-.02em;color:#e8ecf1;line-height:1}
.cl-guia-hero-title span {color:#e74c3c}
.cl-guia-hero-tag {font-size:13px;color:#8b95a5;margin-top:6px}
.cl-guia-stats {display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:18px 0}
.cl-guia-stat {text-align:center;padding:16px;border-radius:16px;background:linear-gradient(145deg, rgba(52,152,219,.10), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);transition:transform .2s ease,border-color .2s ease}
.cl-guia-stat:hover {transform:translateY(-3px);border-color:#3498db;box-shadow:0 0 18px rgba(52,152,219,.15)}
.cl-guia-stat-val {font-size:28px;font-weight:700;background:linear-gradient(90deg,#3498db,#5dade2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.cl-guia-stat-lbl {font-size:11px;color:#8b95a5;text-transform:uppercase;letter-spacing:.08em;margin-top:6px}
.cl-guia-section-title {font-size:18px;font-weight:600;margin:24px 0 12px;color:#e8ecf1}
.cl-guia-pipeline {display:flex;align-items:stretch;justify-content:center;gap:8px;margin:18px 0;flex-wrap:wrap}
.cl-guia-node {flex:1 1 140px;min-width:140px;max-width:200px;text-align:center;padding:16px 10px;border-radius:16px;background:linear-gradient(145deg, rgba(52,152,219,.10), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);position:relative;transition:transform .2s ease,border-color .2s ease}
.cl-guia-node:hover {transform:translateY(-3px);border-color:#3498db;box-shadow:0 0 18px rgba(52,152,219,.15)}
.cl-guia-node-num {width:28px;height:28px;margin:0 auto 10px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:#fff;background:linear-gradient(135deg,#3498db,#2980b9)}
.cl-guia-node-icon {width:28px;height:28px;margin:0 auto 8px;color:#3498db}
.cl-guia-node-icon svg {width:100%;height:100%}
.cl-guia-node-title {font-size:13px;font-weight:600;color:#e8ecf1}
.cl-guia-node-desc {font-size:11px;color:#8b95a5;margin-top:4px;line-height:1.3}
.cl-guia-badge {display:inline-flex;align-items:center;gap:5px;margin-top:10px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:4px 8px;border-radius:999px}
.cl-guia-badge-ok {background:rgba(46,204,113,.15);color:#2ecc71;border:1px solid rgba(46,204,113,.3)}
.cl-guia-badge-next {background:transparent;color:#f1c40f;border:1px dashed rgba(241,196,15,.5)}
.cl-guia-pulse {width:6px;height:6px;border-radius:50%;background:#2ecc71;box-shadow:0 0 0 0 rgba(46,204,113,.7);animation:cl-guia-pulse 1.8s infinite}
@keyframes cl-guia-pulse {0%{transform:scale(.95);box-shadow:0 0 0 0 rgba(46,204,113,.7)}70%{transform:scale(1);box-shadow:0 0 0 6px rgba(46,204,113,0)}100%{transform:scale(.95);box-shadow:0 0 0 0 rgba(46,204,113,0)}}
.cl-guia-conn {width:24px;height:2px;background:linear-gradient(90deg,#3498db,#1abc9c);align-self:center;opacity:.5;border-radius:1px}
.cl-guia-cards {display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}
.cl-guia-card {padding:18px;border-radius:16px;background:linear-gradient(145deg, rgba(52,152,219,.10), rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);transition:transform .2s ease,border-color .2s ease}
.cl-guia-card:hover {transform:translateY(-3px);border-color:#3498db;box-shadow:0 0 18px rgba(52,152,219,.15)}
.cl-guia-card h4 {margin:0 0 10px;font-size:15px;color:#e8ecf1}
.cl-guia-card ul {margin:0;padding-left:18px;font-size:13px;color:#b0b8c4;line-height:1.5}
.cl-guia-chips {display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}
.cl-guia-chip {font-size:11px;font-weight:500;color:#b0b8c4;padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08)}
.cl-guia-repo {margin:18px 0}
.cl-guia-repo summary {font-size:13px;font-weight:600;color:#3498db;cursor:pointer}
.cl-guia-repo pre {font-size:11px;color:#b0b8c4;background:rgba(0,0,0,.25);padding:12px;border-radius:10px;overflow:auto;border:1px solid rgba(255,255,255,.06)}
.cl-guia-links {display:flex;gap:12px;flex-wrap:wrap;margin-top:18px}
.cl-guia-links a {display:inline-flex;align-items:center;gap:6px;padding:10px 16px;border-radius:12px;font-size:13px;font-weight:600;color:#fff;background:linear-gradient(90deg,#3498db,#2980b9);border:1px solid transparent;transition:transform .2s ease,box-shadow .2s ease}
.cl-guia-links a:hover {transform:translateY(-2px);box-shadow:0 0 16px rgba(52,152,219,.25);text-decoration:none}
.cl-guia-links a.secondary {background:transparent;border-color:rgba(255,255,255,.15);color:#e8ecf1}
.cl-guia-links a.secondary:hover {border-color:#3498db}
.cl-guia-cat-chip {display:inline-block;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#3498db;background:rgba(52,152,219,.10);border:1px solid rgba(52,152,219,.25);padding:5px 10px;border-radius:999px;margin:20px 0 10px}
.cl-guia details.faq {background:linear-gradient(145deg, rgba(255,255,255,.03), rgba(255,255,255,.01));border:1px solid rgba(255,255,255,.06);border-left:3px solid transparent;border-radius:0 12px 12px 0;margin-bottom:10px;overflow:hidden;transition:border-color .18s ease}
.cl-guia details.faq[open] {border-left-color:#3498db}
.cl-guia details.faq summary {list-style:none;cursor:pointer;padding:14px 16px 10px;font-size:14px;font-weight:600;color:#3498db;display:flex;align-items:center;justify-content:space-between}
.cl-guia details.faq summary::-webkit-details-marker {display:none}
.cl-guia details.faq summary::after {content:'▸';color:#8b95a5;transition:transform .18s ease}
.cl-guia details.faq[open]>summary::after {transform:rotate(90deg);color:#3498db}
.cl-guia .faq-simple {padding:10px 16px 12px;margin:6px 0 0;font-size:13px;color:#b0b8c4;line-height:1.6;border-top:1px solid rgba(255,255,255,.06)}
.cl-guia details.tec {margin:0 16px 14px;background:rgba(0,0,0,.25);border-radius:10px;border:1px solid rgba(255,255,255,.06)}
.cl-guia details.tec summary {padding:10px 14px;font-size:12px;color:#3498db;font-weight:600;cursor:pointer;list-style:none}
.cl-guia details.tec summary::-webkit-details-marker {display:none}
.cl-guia details.tec summary::after {content:'＋';margin-left:6px;transition:transform .18s ease}
.cl-guia details.tec[open]>summary::after {content:'−'}
.cl-guia .tec-body {padding:0 14px 12px;font-size:12px;color:#9aa3b2;line-height:1.5}
.cl-guia .tec-body code {font-family:SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;background:rgba(255,255,255,.08);padding:1px 5px;border-radius:4px;color:#e8ecf1;font-size:11px}
.cl-guia [role="tablist"], .cl-guia [data-baseweb="tab-list"] {border-bottom:none;margin-bottom:16px}
.cl-guia [role="tablist"] button, .cl-guia [data-baseweb="tab-list"] button {background:transparent;border:1px solid rgba(255,255,255,.08);border-radius:999px;color:#8b95a5;padding:8px 16px;margin:0 6px 8px 0;font-size:13px;font-weight:500;transition:all .2s ease}
.cl-guia [role="tablist"] button[aria-selected="true"], .cl-guia [data-baseweb="tab-list"] button[aria-selected="true"] {background:linear-gradient(90deg,#3498db,#2980b9);color:#fff;border-color:transparent;box-shadow:0 0 12px rgba(52,152,219,.25)}
@media (prefers-reduced-motion: reduce) { .cl-guia-pulse, .cl-guia-stat, .cl-guia-node, .cl-guia-card, .cl-guia details.faq summary::after, .cl-guia details.tec summary::after {animation:none;transition:none} }
@media (max-width: 640px) {
  .cl-guia-hero {flex-direction:column;align-items:flex-start;text-align:left}
  .cl-guia-hero img {width:52px;height:65px}
  .cl-guia-hero-title {font-size:26px}
  .cl-guia-pipeline {flex-direction:column;align-items:center}
  .cl-guia-node {max-width:100%;width:100%}
  .cl-guia-conn {width:2px;height:24px;background:linear-gradient(180deg,#3498db,#1abc9c)}
  .cl-guia-stats {grid-template-columns:repeat(2,1fr)}
}
</style>
""")


def _guia_estructura_html() -> str:
    stats = "".join(
        f"""
        <div class="cl-guia-stat">
          <div class="cl-guia-stat-val">{s['valor']}</div>
          <div class="cl-guia-stat-lbl">{s['etiqueta']}</div>
        </div>
        """
        for s in _GUIA_STATS
    )
    pipeline_parts = []
    for i, n in enumerate(_GUIA_PIPELINE):
        ok = n["estado"] == "Operando hoy"
        badge_cls = "cl-guia-badge-ok" if ok else "cl-guia-badge-next"
        pulse = '<span class="cl-guia-pulse"></span>' if ok else ""
        pipeline_parts.append(_guia_block(f"""
        <div class="cl-guia-node">
          <div class="cl-guia-node-num">{n['num']}</div>
          <div class="cl-guia-node-icon">{n['icono']}</div>
          <div class="cl-guia-node-title">{n['titulo']}</div>
          <div class="cl-guia-node-desc">{n['desc']}</div>
          <div class="cl-guia-badge {badge_cls}">{pulse}{n['estado']}</div>
        </div>
        """))
        if i < len(_GUIA_PIPELINE) - 1:
            pipeline_parts.append('<div class="cl-guia-conn"></div>')
    pipeline = "".join(pipeline_parts)
    vistas = "".join(
        _guia_block(f"""
        <div class="cl-guia-card">
          <h4>{v['titulo']}</h4>
          <ul>
            {''.join(f'<li>{it}</li>' for it in v['items'])}
          </ul>
        </div>
        """)
        for v in _GUIA_VISTAS
    )
    chips = "".join(f'<span class="cl-guia-chip">{c}</span>' for c in _GUIA_TECH_CHIPS)
    return _guia_block(f"""
    <div class="cl-guia">
      <div class="cl-guia-hero">
        <img src="data:image/svg+xml;base64,{_LOGO_B64}" alt="CentineLA logo">
        <div>
          <div class="cl-guia-hero-title">Centine<span>LA</span></div>
          <div class="cl-guia-hero-tag">Alerta temprana comunitaria para la Quebrada La Honda · datos abiertos · bajo costo</div>
        </div>
      </div>
      <div class="cl-guia-stats">
        {stats}
      </div>
      <div class="cl-guia-section-title">Cómo fluye la información</div>
      <div class="cl-guia-pipeline">
        {pipeline}
      </div>
      <div class="cl-guia-section-title">Dos formas de ver el sistema</div>
      <div class="cl-guia-cards">
        {vistas}
      </div>
      <div class="cl-guia-section-title">Tecnologías</div>
      <div class="cl-guia-chips">
        {chips}
      </div>
      <details class="cl-guia-repo">
        <summary>Estructura del repositorio</summary>
        <pre>{_GUIA_REPO}</pre>
      </details>
      <div class="cl-guia-links">
        <a href="https://github.com/B41r0n/CentineLA" target="_blank" rel="noopener">Ver código en GitHub</a>
        <a class="secondary" href="{_GUIA_PROD_URL}" target="_blank" rel="noopener">Abrir el tablero</a>
      </div>
    </div>
    """)


def _guia_faq_html() -> str:
    sections = []
    current_cat = None
    for item in _GUIA_FAQ:
        if item["cat"] != current_cat:
            current_cat = item["cat"]
            sections.append(f'<div class="cl-guia-cat-chip">{current_cat}</div>')
        sections.append(_guia_block(f"""
        <details class="faq">
          <summary><span>{item['q']}</span></summary>
          <div class="faq-simple">{item['simple']}</div>
          <details class="tec">
            <summary>Ver detalle técnico</summary>
            <div class="tec-body">{item['tecnico']}</div>
          </details>
        </details>
        """))
    secciones_html = "\n".join(sections)
    return _guia_block(f"""
    <div class="cl-guia">
      {secciones_html}
    </div>
    """)


@st.dialog("CentineLA · Guía del proyecto", width="large")
def dialog_guia() -> None:
    st.markdown(_GUIA_CSS, unsafe_allow_html=True)
    tabs = st.tabs(["🧭 Estructura del proyecto", "💬 Preguntas frecuentes"])
    with tabs[0]:
        st.markdown(_guia_estructura_html(), unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(_guia_faq_html(), unsafe_allow_html=True)


# ── estilos globales ───────────────────────────────────────────────────────────

CSS_GLOBAL = """
<style>
:root{--cl-bg:#0b0d11;--cl-card:#13171d;--cl-border:rgba(255,255,255,.08);
--cl-text:#e8ecf1;--cl-muted:#8b95a5;--cl-accent:#3498db;
--cl-ok:#2ecc71;--cl-warn:#f1c40f;--cl-danger:#e74c3c}
[data-testid="stMainBlockContainer"]{padding-top:4rem;max-width:1400px}
[data-testid="stSidebarUserContent"]{padding-top:.25rem;overflow:visible}
[data-testid="stSidebarHeader"]{padding-bottom:0}
[data-testid="stMarkdownContainer"]{overflow:visible}
[data-testid="stMarkdown"]{overflow:visible}
h2,h3{font-weight:600!important;letter-spacing:-.01em}
.cl-card{background:var(--cl-card);border:1px solid var(--cl-border);border-radius:16px;padding:18px 20px}
[class*="st-key-card-"]{background:var(--cl-card);border-radius:16px}
div[data-testid="stHorizontalBlock"]:has(.st-key-card-hero){align-items:stretch}
div[data-testid="stHorizontalBlock"]:has(.st-key-card-hero) [data-testid="column"]{display:flex}
div[data-testid="stHorizontalBlock"]:has(.st-key-card-hero) [data-testid="column"]>div{width:100%}
div[data-testid="stHorizontalBlock"]:has(.st-key-card-hero) [data-testid="stVerticalBlockBorderWrapper"]{height:100%;display:flex;flex-direction:column}
div[data-testid="stHorizontalBlock"]:has(.st-key-card-hero) [data-testid="stVerticalBlockBorderWrapper"]>div{flex:1}
.cl-chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 16px}
.cl-chip{background:var(--cl-card);border:1px solid var(--cl-border);border-radius:12px;padding:10px 14px}
.cl-chip .k{font-size:11px;color:var(--cl-muted);text-transform:uppercase;letter-spacing:.06em}
.cl-chip .v{font-size:20px;font-weight:600;margin-top:2px}
.cl-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:500;white-space:nowrap}
.cl-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.cl-hero{background:var(--cl-card);border:1px solid var(--cl-border);border-left:4px solid var(--c);border-radius:16px;padding:22px 24px}
.cl-hero .title{font-size:clamp(20px,3vw,30px);font-weight:700;letter-spacing:.02em;margin:8px 0 4px}
.cl-hero .action{color:var(--cl-muted);font-size:16px}
.cl-gauge{position:relative;display:flex;height:10px;margin:18px 0 6px}
.cl-gauge .mk{position:absolute;top:-4px;width:4px;height:18px;border-radius:2px;background:#fff;transform:translateX(-50%)}
.cl-gauge-lbl{position:relative;height:16px;font-size:11px;color:var(--cl-muted)}
.cl-gauge-lbl>span{position:absolute;transform:translateX(-50%)}
.cl-gauge-lbl>span:first-child{transform:none}
.cl-gauge-lbl>span:last-child{transform:translateX(-100%)}
.cl-grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:8px}
.cl-item{display:grid;grid-template-columns:64px auto 1fr 46px;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid var(--cl-border)}
.cl-bar{height:6px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}
.cl-bar>span{display:block;height:100%;border-radius:999px}
.cl-stat .k{font-size:12px;color:var(--cl-muted);text-transform:uppercase;letter-spacing:.06em}
.cl-stat .v{font-size:clamp(26px,4vw,36px);font-weight:600;line-height:1.1;margin:4px 0}
.cl-stat .u{font-size:12px;color:var(--cl-muted)}
[data-testid="stMetric"]{background:var(--cl-card);border:1px solid var(--cl-border);border-radius:12px;padding:12px 16px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
[data-testid="stExpander"] details{border-radius:12px!important;border:1px solid var(--cl-border)!important;background:var(--cl-card)}
.st-key-btn-guia{margin-bottom:8px}
.st-key-btn-guia button{width:100%;background:rgba(52,152,219,.08);border:1px solid #3498db;color:#e8ecf1;border-radius:12px;padding:10px 16px;font-weight:500;transition:all .2s ease}
.st-key-btn-guia button:hover{background:rgba(52,152,219,.18);box-shadow:0 0 12px rgba(52,152,219,.25)}
/* ── mobile (≤640px): vista principal ── */
@media (max-width: 640px){
/* header principal */
.cl-header-main .cl-header-row{flex-wrap:wrap}
.cl-header-main .cl-header-title{font-size:30px!important;white-space:normal!important}
.cl-header-main img{width:36px;height:46px}
.cl-header-main .cl-pill{white-space:normal}
/* barra lateral / brand */
.cl-sidebar-brand .cl-sidebar-title{font-size:30px!important}
.cl-sidebar-brand img{width:56px;height:70px}
/* calendario 90 días: 15 columnas y sin alto mínimo para que no desborde */
.cl-cal{grid-template-columns:repeat(15,1fr)!important;gap:3px!important}
.cl-cal>div{min-height:0!important}
/* leyenda del calendario */
.cl-leyenda{flex-wrap:wrap;gap:8px 16px!important}
/* tarjetas y chips */
.cl-card{padding:14px}
.cl-chip .v{font-size:18px}
.cl-stat .v{font-size:24px}
}
</style>
"""

# ── logo SVG en base64 ──────────────────────────────────────────────────────────
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="44" height="56" viewBox="0 0 44 56">'
    '<path d="M22 2 C22 2 4 24 4 38 C4 48 12 54 22 54 C32 54 40 48 40 38 C40 24 22 2 22 2 Z" fill="#3498db"/>'
    '<ellipse cx="16" cy="26" rx="4" ry="8" fill="#ffffff" opacity="0.30" transform="rotate(-18 16 26)"/>'
    '</svg>'
)
_LOGO_B64 = base64.b64encode(_LOGO_SVG.encode("utf-8")).decode("ascii")


def _pill(estado: str, texto: str | None = None) -> str:
    """Pill de estado con su color (texto: por defecto, la etiqueta visible del estado)."""
    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    return (
        f'<span class="cl-pill" style="background:{ui["soft"]};color:{ui["pill_txt"]}">'
        f'<span class="cl-dot" style="background:{ui["color"]}"></span>{texto or ui["etiqueta"]}</span>'
    )


def _badge_estado(estado: str) -> str:
    """Pill de estado con texto 'Prototipo activo · {etiqueta del estado}'."""
    return _pill(estado, f"Prototipo activo · {ESTADO_UI.get(estado, ESTADO_UI['NORMAL'])['etiqueta']}")


def _badge_fijo(texto: str, bg: str, color: str) -> str:
    return (
        f'<span class="cl-pill" style="background:{bg};color:{color}">{texto}</span>'
    )


def render_header(estado: str, ultima_lectura: str) -> None:
    """Renderiza el header con logo, título, subtítulos y badges (pills)."""
    st.markdown(
        f"""
<div class="cl-header-main" style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;overflow:visible">
  <div class="cl-header-row" style="display:flex;align-items:center;gap:8px;overflow:visible">
    <img src="data:image/svg+xml;base64,{_LOGO_B64}" width="44" height="56" style="flex-shrink:0;object-fit:contain;display:block;vertical-align:middle" alt="CentineLA logo">
    <div class="cl-header-title" style="font-size:44px;font-weight:500;letter-spacing:-1px;color:#3498db;line-height:1;white-space:nowrap;flex-shrink:1;min-width:0">
      Centine<span style="color:#e74c3c">LA</span>
    </div>
  </div>
  <hr style="margin:4px 0;border:none;border-top:0.5px solid rgba(128,128,128,0.35)">
  <div style="font-size:16px;font-weight:500">Sistema integrado de vigilancia comunitaria</div>
  <div style="font-size:14px;color:var(--cl-muted)">Quebrada La Honda · Comuna 4 · Medellín</div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:4px">
    {_badge_estado(estado)}
    {_badge_fijo("Territorio INN 2026 · Reto #7", "rgba(52,152,219,.14)", "#6cb4ee")}
    {_badge_fijo(f"Datos al {ultima_lectura}", "rgba(255,255,255,.06)", "var(--cl-muted)")}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CentineLA — Vigilancia comunitaria · Q. La Honda",
        page_icon="💧",
        layout="wide",
    )
    st.markdown(CSS_GLOBAL, unsafe_allow_html=True)

    # Cargar credenciales desde Streamlit secrets si existen (antes de cualquier uso)
    try:
        for k in ("SODAPY_APP_TOKEN", "SODAPY_USERNAME", "SODAPY_PASSWORD"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:  # noqa: S110 — sin secrets.toml (uso local): se usan .env / variables de entorno
        pass

    # Sembrar BD SQLite si no existe (lazy; es solo una copia, no debe frenar el tablero)
    try:
        asegurar_db_sembrada()
    except Exception as exc:
        print(f"[dashboard] no se pudo crear centinela.db: {exc!r}", file=sys.stderr)

    with st.spinner("Cargando datos..."):
        try:
            log = cargar_log()
        except FileNotFoundError:
            st.error(f"No se encontró el log de gateway en: `{LOG_PATH}`")
            st.stop()

        try:
            proxy = cargar_proxy()
        except FileNotFoundError:
            st.error(f"No se encontró el proxy en: `{PROXY_PATH}`")
            st.stop()

        try:
            modelo = cargar_modelo()
        except FileNotFoundError:
            st.warning(f"Modelo no encontrado en `{MODEL_PATH}`. Vista Operador limitada.")
            modelo = None

    # Header con logo y badges
    ultima = log.iloc[-1]
    estado_actual = str(ultima["estado"])
    ultima_fecha_str = _fecha_es(ultima["timestamp"], "%d-%b-%Y")
    render_header(estado_actual, ultima_fecha_str)

    # Apartado "¿De qué trata CentineLA?" — visible en ambas vistas
    with st.expander("¿De qué trata CentineLA?", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**El problema**")
            st.write(
                "Las crecientes súbitas de la quebrada pueden formarse en pocas horas "
                "tras lluvias intensas en la cuenca, con poco tiempo de reacción para "
                "las familias cercanas."
            )
        with col2:
            st.markdown("**Cómo funciona**")
            st.write(
                "Toma datos de lluvia de estaciones del IDEAM, estima cuánta agua "
                "escurre hacia la quebrada y un modelo de aprendizaje automático calcula "
                "la probabilidad de riesgo en las próximas 6 horas: que la lluvia "
                "acumulada llegue al punto en que el terreno ya no la absorbe y el agua "
                "escurre hacia la quebrada. No mide el nivel del agua."
            )
        with col3:
            st.markdown("**Qué entrega**")
            st.write(
                "Un semáforo claro para la comunidad (verde, amarillo, rojo) y un panel "
                "técnico para la JAC con el detalle de cada lectura."
            )
        st.caption(
            "Prototipo: este panel usa el histórico del IDEAM procesado y un gateway "
            "simulado. Proyecto individual de Bairon Nicolás Calle Rivera · ITM · "
            "Territorio INN 2026, Reto #7."
        )

    # Sidebar
    # Logo + título en sidebar (grande, arriba a la izquierda)
    st.sidebar.markdown(
        f"""
<div class="cl-sidebar-brand" style="display:flex;align-items:center;gap:10px;margin-bottom:6px;justify-content:flex-start;overflow:visible">
  <img src="data:image/svg+xml;base64,{_LOGO_B64}" width="76" height="95" style="flex-shrink:0;object-fit:contain;display:block;vertical-align:middle" alt="CentineLA logo">
  <div class="cl-sidebar-title" style="font-size:38px;font-weight:500;letter-spacing:-1px;color:#3498db;line-height:1;white-space:nowrap;flex-shrink:1;min-width:0">
    Centine<span style="color:#e74c3c">LA</span>
  </div>
</div>
<hr style="margin:4px 0;border:none;border-top:0.5px solid rgba(128,128,128,0.35)">
""",
        unsafe_allow_html=True,
    )
    vista = st.sidebar.radio(
        "Seleccionar vista",
        ["Vista Pública", "Vista Operador (JAC)"],
        format_func=lambda x: "🏘️ Vista pública" if x == "Vista Pública" else "🛠️ Vista operador (JAC)",
    )
    st.sidebar.caption(f"Última lectura: {ultima['timestamp']}")
    st.sidebar.caption(f"Estado: {ESTADO_UI[estado_actual]['etiqueta']}")
    st.sidebar.caption("CentineLA — Bairon Nicolás Calle Rivera · ITM · Territorio INN 2026")

    # Botón guía del proyecto en sidebar
    st.sidebar.divider()
    if st.sidebar.button("📖 Estructura y preguntas frecuentes", key="btn-guia", width="stretch"):
        dialog_guia()

    # Botón "Actualizar ahora" en sidebar
    if st.sidebar.button("🔄 Actualizar ahora", width="stretch"):
        est = _estado_actualizacion()
        restante = est["proximo"] - time.time()
        if restante > 0:
            st.session_state["msg_actualizar"] = (
                "info",
                f"Actualización reciente. Puedes intentar de nuevo en {int(restante // 60) + 1} min.",
                "",
            )
        elif not est["lock"].acquire(blocking=False):
            st.session_state["msg_actualizar"] = (
                "info",
                "Ya hay una actualización en curso. Espera unos segundos.",
                "",
            )
        else:
            ok = False
            msg_tipo, msg_texto, msg_caption = "error", "No se pudo actualizar en este momento. Se mantienen los últimos datos guardados.", ""
            try:
                with st.spinner("Consultando IDEAM y recalculando..."):
                    resultado = actualizar_ahora()
                if resultado["ok"]:
                    ok = True
                    total_nuevas = sum(resultado["filas_nuevas_por_estacion"].values())
                    ultimas_est = resultado.get("ultima_lectura_por_estacion", {})
                    nombres_est = {
                        "0027015310": "Metromedellín",
                        "0027015330": "Olaya Herrera",
                    }
                    partes = []
                    for cod, nombre in nombres_est.items():
                        ts = ultimas_est.get(cod, "")
                        if ts:
                            partes.append(f"{nombre} hasta {ts}")
                    msg_transparencia = ""
                    if len(partes) >= 2:
                        msg_transparencia = " · ".join(partes) + ". El tablero usa la última hora con al menos 2 estaciones."

                    if total_nuevas == 0:
                        msg_tipo, msg_texto = (
                            "info",
                            f"Sin datos nuevos en IDEAM — última disponible: {resultado['ultima_lectura']}",
                        )
                    else:
                        msg_tipo, msg_texto = (
                            "success",
                            f"{total_nuevas} lecturas nuevas — última: {resultado['ultima_lectura']}",
                        )
                    msg_caption = msg_transparencia
                else:
                    msg_caption = resultado["error"]
            except Exception as exc:  # capturar todo es intencional
                # Error no previsto: detalle a los logs del servidor, al usuario solo
                # el mensaje genérico (nunca una traza).
                print(f"[actualizar_ahora] error no controlado: {exc!r}", file=sys.stderr)
            finally:
                est["lock"].release()
                est["proximo"] = time.time() + (COOLDOWN_OK_S if ok else COOLDOWN_ERR_S)

            st.session_state["msg_actualizar"] = (msg_tipo, msg_texto, msg_caption)
            st.cache_data.clear()
            st.rerun()

    if "msg_actualizar" in st.session_state:
        tipo, texto, caption = st.session_state.pop("msg_actualizar")
        if tipo == "success":
            st.sidebar.success(texto)
        elif tipo == "info":
            st.sidebar.info(texto)
        else:
            st.sidebar.error(texto)
        if caption:
            st.sidebar.caption(caption)

    if vista == "Vista Pública":
        vista_publica(log, proxy)
    else:
        if modelo is None:
            st.error("clf_6h.joblib no disponible — Vista Operador no puede mostrar importancias.")
        else:
            vista_operador(log, proxy, modelo)

    # Footer
    st.divider()
    st.caption(
        "CentineLA · Bairon Nicolás Calle Rivera · ITM · Territorio INN 2026 · "
        "Datos: IDEAM (datos.gov.co) · Prototipo con gateway simulado"
    )


if __name__ == "__main__":
    main()
