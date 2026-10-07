"""
CentineLA — Dashboard Streamlit
Vista Pública + Vista Operador (JAC)

Uso:
    streamlit run dashboard/app.py
"""

import importlib.util
import base64
import io
import math
import os
import sys
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

# Cargar simulate/11_actualizar_ahora.py via importlib (nombre con número)
_ACTUALIZAR_PATH = BASE_DIR / "simulate" / "11_actualizar_ahora.py"
_act_spec = importlib.util.spec_from_file_location("actualizar_ahora", _ACTUALIZAR_PATH)
_act_mod = importlib.util.module_from_spec(_act_spec)
_act_spec.loader.exec_module(_act_mod)
actualizar_ahora = _act_mod.actualizar_ahora
asegurar_db_sembrada = _act_mod._asegurar_db_sembrada

# ── umbrales ternarios del sistema de alertas ──
UMBRAL_PRECAUCION = 0.30
UMBRAL_ALERTA = 0.70

# ── tokens de diseño por estado (única fuente) ──
ESTADO_UI = {
    "NORMAL": {
        "color": "#2ecc71",
        "pill_txt": "#34d27b",
        "soft": "rgba(46,204,113,.14)",
        "banner": "NORMAL",
        "accion": "No hay señales de alerta. Monitoreo activo.",
    },
    "PRECAUCIÓN": {
        "color": "#f39c12",
        "pill_txt": "#f5b041",
        "soft": "rgba(243,156,18,.14)",
        "banner": "PRECAUCIÓN — LLUVIAS SOSTENIDAS",
        "accion": "Lluvias sostenidas detectadas. Mantenerse atentos.",
    },
    "ALERTA": {
        "color": "#e74c3c",
        "pill_txt": "#ff6b5e",
        "soft": "rgba(231,76,60,.16)",
        "banner": "ALERTA — RIESGO EN LAS PRÓXIMAS 6 HORAS",
        "accion": "Riesgo de creciente en las próximas 6 horas. Activar protocolo de la JAC.",
    },
}


def clasificar_estado(proba: float) -> str:
    """Sistema ternario de alertas."""
    if proba >= UMBRAL_ALERTA:
        return "ALERTA"
    elif proba >= UMBRAL_PRECAUCION:
        return "PRECAUCIÓN"
    return "NORMAL"

# ── anchors geográficos de la quebrada (dato-duro AMVA + referencia visual) ─
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
    "proba_alerta": "Probabilidad de creciente",
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
    Arví→Jardín Botánico y Jardín Botánico→Moravia, interpolando linealmente
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


def check_red(timeout: int = 3) -> bool:
    """Verifica conectividad a los tiles de OSM con timeout corto."""
    try:
        import requests
        requests.get("https://tile.openstreetmap.org/0/0/0.png", timeout=timeout)
        return True
    except Exception:
        return False


def _compacto(html: str) -> str:
    """Elimina líneas en blanco y espacios sobrantes de HTML."""
    return "\n".join(l for l in html.split("\n") if l.strip())


# ── carga de datos con caché ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_log() -> pd.DataFrame:
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Normalizar a sistema ternario (el log histórico puede venir binario)
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
    st.caption(
        "Mapa interactivo (OpenStreetMap, tema oscuro). "
        "Posiciones de nodos son ilustrativas — no GPS de campo."
    )


def _mapa_estatico(anclas: list[dict], nodos: list[dict], estado: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

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
    handle_estado = mpatches.Patch(color=color, label=estado)
    handle_ancla = mlines.Line2D(
        [], [], marker="o", color="w",
        markerfacecolor="white", markeredgecolor="#3498db",
        markersize=10, markeredgewidth=2.5, label="Ancla geográfica",
    )
    leg = ax.legend(handles=[handle_estado, handle_ancla], loc="lower right", fontsize=8,
                    facecolor="#13171d", edgecolor="#232830", labelcolor="#e8ecf1")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="#13171d")
    plt.close(fig)
    buf.seek(0)
    st.image(
        buf.getvalue(),
        caption="⚠️ Mapa estático (sin conexión detectada). Posiciones ilustrativas, no GPS de campo.",
        use_container_width=True,
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


# ── helpers de Vista Pública ──────────────────────────────────────────────────

def _color_proba(proba: float) -> str:
    if proba >= UMBRAL_ALERTA:
        return "#e74c3c"
    if proba >= UMBRAL_PRECAUCION:
        return "#f39c12"
    return "#2ecc71"


def _gauge_html(proba: float) -> str:
    """Gauge con tres zonas (verde/amarillo/rojo) y marcador."""
    p = max(0.0, min(1.0, proba))
    w_green = UMBRAL_PRECAUCION * 100
    w_yellow = (UMBRAL_ALERTA - UMBRAL_PRECAUCION) * 100
    w_red = (1.0 - UMBRAL_ALERTA) * 100
    return f"""
<div class="cl-gauge">
  <div style="width:{w_green}%;background:rgba(46,204,113,.35);border-radius:999px 0 0 999px;"></div>
  <div style="width:{w_yellow}%;background:rgba(243,156,18,.35);"></div>
  <div style="width:{w_red}%;background:rgba(231,76,60,.35);border-radius:0 999px 999px 0;"></div>
  <div class="cl-mk" style="left:{p*100}%"></div>
</div>
<div class="cl-gauge-lbl">
  <span>0%</span><span>30%</span><span>70%</span><span>100%</span>
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
      <div class="k">Probabilidad de creciente</div>
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
    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    color_p = _color_proba(proba)
    return _compacto(f"""
<div class="cl-card" style="text-align:center;display:flex;flex-direction:column;align-items:center;gap:10px">
  <div>{_pill(estado)}</div>
  <div style="font-size:14px;color:var(--cl-muted)">Probabilidad de creciente</div>
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

    # Estado máximo por día: 0=NORMAL, 1=PRECAUCIÓN, 2=ALERTA
    estado_map = {"NORMAL": 0, "PRECAUCIÓN": 1, "ALERTA": 2}
    colores = {0: "#1f6f4a", 1: "#f39c12", 2: "#e74c3c"}  # verde apagado para NORMAL
    dia_estado = (
        log.groupby(log["timestamp"].dt.date)["estado"]
        .apply(lambda s: max((estado_map.get(e, 0) for e in s), default=0))
    )

    celdas = []
    for dia in rango:
        d = dia.date()
        nivel = int(dia_estado.get(d, 0))
        color = colores[nivel]
        celda = (
            '<div title="' + str(d) + '" style="background:' + color
            + ';border-radius:4px;aspect-ratio:1;min-height:18px"></div>'
        )
        celdas.append(celda)

    grid_html = (
        '<div style="display:grid;grid-template-columns:repeat(18,1fr);'
        'gap:4px;max-width:100%;margin-bottom:8px">'
        + "".join(celdas)
        + "</div>"
    )
    cuadro_n = (
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:#1f6f4a;border-radius:2px;vertical-align:middle"></span>'
    )
    cuadro_p = (
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:#f39c12;border-radius:2px;vertical-align:middle"></span>'
    )
    cuadro_a = (
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:#e74c3c;border-radius:2px;vertical-align:middle"></span>'
    )
    leyenda = (
        '<div style="display:flex;gap:16px;font-size:12px;color:var(--cl-muted)">'
        + cuadro_n + " Normal&nbsp;&nbsp;"
        + cuadro_p + " Precaución&nbsp;&nbsp;"
        + cuadro_a + " Alerta"
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
    """Calcula lluvia acumulada en ventanas deslizantes desde la ultima fecha disponible.

    Usa proxy['timestamp'].max() como referencia — NUNCA la fecha real del sistema.
    Devuelve (ult_24h_mm, ult_30d_mm, anio_mm, ref_fecha_str).

    Se usan ventanas deslizantes (24h, 30d) en vez de dia/mes del calendario porque
    el historico termina en 2026-07-09 con P_basin genuinamente en 0 para julio
    (no llovio esos dias, verificado contra datos crudos). El acumulado de 30 dias
    y anual reflejan mejor el contexto reciente sin mostrar 0 de forma confusa.
    """
    ultima_ts = proxy["timestamp"].max()   # referencia fija al dataset, no al reloj
    anio_ref = ultima_ts.year

    p = proxy.dropna(subset=["P_basin"])

    ult_24h = float(p[p["timestamp"] >= ultima_ts - pd.Timedelta(hours=24)]["P_basin"].sum())
    ult_30d = float(p[p["timestamp"] >= ultima_ts - pd.Timedelta(days=30)]["P_basin"].sum())
    anio_mm = float(p[p["timestamp"].dt.year == anio_ref]["P_basin"].sum())
    ref_str = str(ultima_ts.date())
    return ult_24h, ult_30d, anio_mm, ref_str


# ── vista pública ─────────────────────────────────────────────────────────────

def vista_publica(log: pd.DataFrame, proxy: pd.DataFrame) -> None:
    ultima = log.iloc[-1]
    estado_actual = str(ultima["estado"])
    proba_actual = float(ultima["proba_alerta"])
    ultima_ts_str = ultima["timestamp"].strftime("%d-%b-%Y %H:%M").replace(
        "Jan", "ene"
    ).replace("Feb", "feb").replace("Mar", "mar").replace("Apr", "abr").replace(
        "May", "may"
    ).replace("Jun", "jun").replace("Jul", "jul").replace("Aug", "ago").replace(
        "Sep", "sep"
    ).replace("Oct", "oct").replace("Nov", "nov").replace("Dec", "dic")

    # 1. Fila de chips con datos reales (ancho completo)
    ult_24h, ult_30d, anio_mm, _ = _lluvia_acumulada(proxy)
    # Días con alerta en últimos 30 días
    corte_30d = ultima["timestamp"] - pd.Timedelta(days=30)
    log_30d = log[log["timestamp"] >= corte_30d]
    dias_alerta_30d = log_30d[log_30d["estado"] == "ALERTA"]["timestamp"].dt.date.nunique()

    chips = [
        ("Estado", _pill(estado_actual)),
        ("Probabilidad", f"{proba_actual:.1%}"),
        ("Lluvia 24 h", f"{ult_24h:.1f} mm"),
        ("Última lectura", ultima_ts_str),
        ("Días con alerta (30 d)", str(dias_alerta_30d)),
    ]
    st.markdown(_chips_html(chips, estado=estado_actual), unsafe_allow_html=True)

    # 2. Dos columnas: Hero (izq) + Últimos días con alertas (der)
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        with st.container(border=True, key="card-hero"):
            st.markdown(_hero_html(estado_actual, proba_actual, ultima_ts_str), unsafe_allow_html=True)

    with col_der:
        with st.container(border=True, key="card-alertas"):
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
                    fecha_str = d.strftime("%d-%b").replace(
                        "Jan", "ene"
                    ).replace("Feb", "feb").replace("Mar", "mar").replace("Apr", "abr").replace(
                        "May", "may"
                    ).replace("Jun", "jun").replace("Jul", "jul").replace("Aug", "ago").replace(
                        "Sep", "sep"
                    ).replace("Oct", "oct").replace("Nov", "nov").replace("Dec", "dic")
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
            "ilustrativas, no son GPS de campo."
        )

    # 4. Lluvia acumulada: 3 tarjetas con _tarjeta_html en grid
    st.subheader("Lluvia acumulada")
    st.markdown(
        _compacto(f'<div class="cl-grid3">'
        + _tarjeta_html("🌧️", "Últimas 24 horas", f"{ult_24h:.1f}", "mm")
        + _tarjeta_html("📅", "Últimos 30 días", f"{ult_30d:.1f}", "mm")
        + _tarjeta_html("📆", "Año en curso", f"{anio_mm:.1f}", "mm")
        + '</div>'),
        unsafe_allow_html=True,
    )
    st.caption(
        "Cuánta lluvia ha caído en la cuenca. Más lluvia acumulada significa suelo "
        "más saturado y mayor riesgo."
    )

    # 5. Historial de los últimos 90 días
    st.subheader("Historial de los últimos 90 días")
    with st.container(border=True, key="card-calendario"):
        st.markdown(_calendario_html(log, n_dias=90), unsafe_allow_html=True)
    st.caption(
        "Cada cuadro es un día. Verde: sin alertas. Amarillo: hubo "
        "precaución. Rojo: hubo alerta."
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
                mode="markers", name="Precaución",
                marker=dict(color="#f39c12", size=6, symbol="x"), yaxis="y2",
            ))
        if not alertas.empty:
            fig.add_trace(go.Scatter(
                x=alertas["timestamp"], y=alertas["proba_alerta"],
                mode="markers", name="Alerta",
                marker=dict(color="#e74c3c", size=6, symbol="x"), yaxis="y2",
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"En este período: {len(alertas)} alertas y {len(precauciones)} precauciones. "
            "Las líneas verticales se muestran submuestreadas para rendimiento."
        )


# ── vista operador ────────────────────────────────────────────────────────────

def vista_operador(log: pd.DataFrame, proxy: pd.DataFrame, modelo) -> None:
    st.header("🛠️ Vista operador (JAC)")

    # 1. Registro de lecturas recientes
    with st.container(border=True, key="card-registro"):
        st.subheader("📋 Registro de lecturas recientes")
        st.caption(
            "Las últimas 100 lecturas horarias del sistema. Cada fila muestra qué se midió "
            "y qué estado calculó."
        )
        log_mostrar = log.tail(100).rename(columns=NOMBRES_LEGIBLES)
        # Colorear columna Estado de alerta con pandas Styler
        try:
            def color_estado(val):
                if val == "ALERTA":
                    return "color: #ff6b5e; font-weight: 600"
                elif val == "PRECAUCIÓN":
                    return "color: #f5b041; font-weight: 600"
                elif val == "NORMAL":
                    return "color: #34d27b; font-weight: 600"
                return ""
            styled = log_mostrar.style.map(color_estado, subset=["Estado de alerta"])
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(log_mostrar, use_container_width=True)

    # 2. Comportamiento histórico de la cuenca
    _grafico_historico(log, proxy)

    # 3. Confiabilidad del sistema
    with st.container(border=True, key="card-confiabilidad"):
        st.subheader("🎯 Confiabilidad del sistema")
        st.caption(
            "Detectadas = de cada 100 crecientes reales, cuántas avisó el sistema. "
            "Alertas reales = de cada 100 avisos, cuántos fueron crecientes de verdad. "
            "Calculado sobre el periodo de prueba (2025-07-06 → 2026-09-30, 9,051 horas), "
            "que el modelo no vio al entrenar. No se recalcula en vivo — "
            "fuente: simulate/10_metricas_ternario.py."
        )
        RECALL_PRECAUCION = 0.714628
        PRECISION_PRECAUCION = 0.497496
        RECALL_ALERTA = 0.549161
        PRECISION_ALERTA = 0.860902

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                "Crecientes detectadas (desde precaución)",
                f"{RECALL_PRECAUCION:.1%}",
                help="Recall con umbral 0.30 (PRECAUCIÓN). Fuente: simulate/10_metricas_ternario.py",
            )
            st.metric(
                "Crecientes detectadas (solo alerta roja)",
                f"{RECALL_ALERTA:.1%}",
                help="Recall con umbral 0.70 (ALERTA). Fuente: simulate/10_metricas_ternario.py",
            )
        with c2:
            st.metric(
                "Alertas que fueron reales (desde precaución)",
                f"{PRECISION_PRECAUCION:.1%}",
                help="Precisión con umbral 0.30 (PRECAUCIÓN). Fuente: simulate/10_metricas_ternario.py",
            )
            st.metric(
                "Alertas rojas que fueron reales",
                f"{PRECISION_ALERTA:.1%}",
                help="Precisión con umbral 0.70 (ALERTA). Fuente: simulate/10_metricas_ternario.py",
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
            importancias = dict(zip(FEATURE_COLS, modelo.feature_importances_))
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
            st.plotly_chart(fig_imp, use_container_width=True)
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
            "Elegí una fecha y hora para ver qué datos recibió el sistema en ese momento "
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

        if log_dia.empty:
            st.warning("Sin datos en el log para esta fecha.")
            return

        opciones_hora = log_dia["timestamp"].dt.strftime("%H:%M").tolist()
        hora_sel = st.selectbox("Hora (hora local)", opciones_hora, key="hora_inspeccion")

        fila = log_dia[log_dia["timestamp"].dt.strftime("%H:%M") == hora_sel].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Fecha y hora:** `{fila['timestamp']}`")
            st.markdown(f"**Estado de alerta:** `{fila['estado']}`")
            st.markdown(f"**Probabilidad de creciente:** `{fila['proba_alerta']:.4f}`")

        with col_b:
            cols_mostrar = [c for c in ["P_basin", "Q_actual", "Q_lag_1h", "roll_sum_24h", "hora_dia", "mes"]
                            if c in fila.index]
            if cols_mostrar:
                df_vals = pd.DataFrame([{c: round(float(fila[c]), 6) for c in cols_mostrar}])
                df_vals = df_vals.T.rename(columns={0: "Valor"})
                df_vals.index = [NOMBRES_LEGIBLES.get(idx, idx) for idx in df_vals.index]
                st.dataframe(df_vals, use_container_width=True)

    # 6. Simulador
    with st.container(border=True, key="card-simulador"):
        _simulador_modelo(modelo)


# ── simulador interactivo del modelo ─────────────────────────────────────────

_ESCENARIOS = {
    "🌧️ Lluvia fuerte sostenida": {
        "sim_lag_1h": 15.0, "sim_lag_3h": 35.0, "sim_lag_6h": 45.0,
        "sim_roll12": 40.0, "sim_roll24": 45.0, "sim_roll48": 48.0,
        "sim_Q": 3.0,
    },
    "⛈️ Aguacero puntual": {
        "sim_lag_1h": 25.0, "sim_lag_3h": 15.0, "sim_lag_6h": 5.0,
        "sim_roll12": 5.0, "sim_roll24": 5.0, "sim_roll48": 2.0,
        "sim_Q": 1.0,
    },
    "☀️ Normal": {
        "sim_lag_1h": 0.0, "sim_lag_3h": 0.0, "sim_lag_6h": 0.0,
        "sim_roll12": 0.0, "sim_roll24": 0.0, "sim_roll48": 0.0,
        "sim_Q": 0.0,
    },
}


def _simulador_modelo(modelo) -> None:
    st.subheader("🧪 Simular un escenario de lluvia")
    st.caption(
        "Ingresá condiciones hipotéticas para ver cómo respondería el sistema. Útil "
        "para entender cuándo se activa cada estado."
    )

    # ── escenarios rápidos ────────────────────────────────────────────────
    cols_btn = st.columns(len(_ESCENARIOS))
    for col, (nombre, vals) in zip(cols_btn, _ESCENARIOS.items()):
        if col.button(nombre, use_container_width=True, key="btn_" + nombre[:6]):
            for k, v in vals.items():
                st.session_state[k] = v

    # ── sliders ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        lag_1h  = st.slider("Lluvia en la última hora (mm)",  0.0, 50.0, step=0.5, key="sim_lag_1h")
        lag_3h  = st.slider("Lluvia en las últimas 3 horas (mm)",  0.0, 50.0, step=0.5, key="sim_lag_3h")
        lag_6h  = st.slider("Lluvia en las últimas 6 horas (mm)",  0.0, 50.0, step=0.5, key="sim_lag_6h")
        q_act   = st.number_input("Escorrentía actual (mm)", min_value=0.0, max_value=20.0, step=0.1, key="sim_Q")
    with col2:
        roll12  = st.slider("Lluvia acumulada 12 h (mm)", 0.0, 50.0, step=0.5, key="sim_roll12")
        roll24  = st.slider("Lluvia acumulada 24 h (mm)", 0.0, 50.0, step=0.5, key="sim_roll24")
        roll48  = st.slider("Lluvia acumulada 48 h (mm)", 0.0, 50.0, step=0.5, key="sim_roll48")

    # ── features derivadas (aproximación) ────────────────────────────────
    ahora = pd.Timestamp.now()
    x_vec = {
        "P_basin":       lag_1h,
        "Q_actual":      q_act,
        "lag_1h":        lag_1h,
        "lag_3h":        lag_3h,
        "lag_6h":        lag_6h,
        "lag_12h":       lag_6h * 0.5,
        "lag_24h":       lag_6h * 0.25,
        "Q_lag_1h":      q_act,
        "Q_lag_3h":      q_act,
        "Q_lag_6h":      q_act,
        "roll_sum_3h":   (lag_1h + lag_3h) * 0.5,
        "roll_sum_6h":   lag_6h,
        "roll_sum_12h":  roll12,
        "roll_sum_24h":  roll24,
        "roll_sum_48h":  roll48,
        "roll_max_6h":   lag_1h,
        "hora_dia":      float(ahora.hour),
        "mes":           float(ahora.month),
    }

    if st.button("Simular", type="primary"):
        x_df = pd.DataFrame([x_vec])[FEATURE_COLS]
        proba = float(modelo.predict_proba(x_df)[0, 1])
        estado_pred = clasificar_estado(proba)
        st.markdown(
            _estado_semaforo_html(estado_pred, proba),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Probabilidad calculada: {proba:.1%}. "
            "Verde menos de 30% · Amarillo 30–70% · Rojo 70% o más. "
            "Los valores intermedios se aproximan a partir de los controles."
        )


# ── estilos globales ───────────────────────────────────────────────────────────

CSS_GLOBAL = """
<style>
:root{--cl-bg:#0b0d11;--cl-card:#13171d;--cl-border:rgba(255,255,255,.08);
--cl-text:#e8ecf1;--cl-muted:#8b95a5;--cl-accent:#3498db;
--cl-ok:#2ecc71;--cl-warn:#f39c12;--cl-danger:#e74c3c}
[data-testid="stMainBlockContainer"]{padding-top:1.5rem;max-width:1400px}
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
.cl-gauge-lbl{display:flex;justify-content:space-between;font-size:11px;color:var(--cl-muted)}
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


def _pill(estado: str) -> str:
    """Genera una pill de estado usando ESTADO_UI."""
    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    return (
        f'<span class="cl-pill" style="background:{ui["soft"]};color:{ui["pill_txt"]}">'
        f'<span class="cl-dot" style="background:{ui["color"]}"></span>{estado}</span>'
    )


def _badge_estado(estado: str) -> str:
    """Genera una pill de estado con texto 'Prototipo activo · {estado}'."""
    ui = ESTADO_UI.get(estado, ESTADO_UI["NORMAL"])
    return (
        f'<span class="cl-pill" style="background:{ui["soft"]};color:{ui["pill_txt"]}">'
        f'<span class="cl-dot" style="background:{ui["color"]}"></span>'
        f'Prototipo activo · {estado}</span>'
    )


def _badge_fijo(texto: str, bg: str, color: str) -> str:
    return (
        f'<span class="cl-pill" style="background:{bg};color:{color}">{texto}</span>'
    )


def render_header(estado: str, ultima_lectura: str) -> None:
    """Renderiza el header con logo, título, subtítulos y badges (pills)."""
    st.markdown(
        f"""
<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:8px;overflow:visible">
  <div style="display:flex;align-items:center;gap:8px;overflow:visible">
    <img src="data:image/svg+xml;base64,{_LOGO_B64}" width="44" height="56" style="flex-shrink:0;object-fit:contain;display:block;vertical-align:middle" alt="CentineLA logo">
    <div style="font-size:44px;font-weight:500;letter-spacing:-1px;color:#3498db;line-height:1;white-space:nowrap;flex-shrink:1;min-width:0">
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
        if "SODAPY_USERNAME" in st.secrets:
            os.environ.setdefault("SODAPY_USERNAME", st.secrets["SODAPY_USERNAME"])
            os.environ.setdefault("SODAPY_PASSWORD", st.secrets["SODAPY_PASSWORD"])
            if "SODAPY_APP_TOKEN" in st.secrets:
                os.environ.setdefault("SODAPY_APP_TOKEN", st.secrets["SODAPY_APP_TOKEN"])
    except Exception:
        pass

    # Sembrar BD SQLite si no existe (lazy, silencioso)
    try:
        asegurar_db_sembrada()
    except Exception:
        pass

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
    ultima_fecha_str = ultima["timestamp"].strftime("%d-%b-%Y").replace(
        "Jan", "ene"
    ).replace("Feb", "feb").replace("Mar", "mar").replace("Apr", "abr").replace(
        "May", "may"
    ).replace("Jun", "jun").replace("Jul", "jul").replace("Aug", "ago").replace(
        "Sep", "sep"
    ).replace("Oct", "oct").replace("Nov", "nov").replace("Dec", "dic")
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
                "la probabilidad de creciente en las próximas 6 horas."
            )
        with col3:
            st.markdown("**Qué entrega**")
            st.write(
                "Un semáforo claro para la comunidad (verde, amarillo, rojo) y un panel "
                "técnico para la JAC con el detalle de cada lectura."
            )
        st.caption(
            "Prototipo: este panel usa el histórico del IDEAM procesado y un gateway "
            "simulado. Proyecto individual de Bairon Nicolas Calle Rivera · ITM · "
            "Territorio INN 2026, Reto #7."
        )

    # Sidebar
    # Logo + título en sidebar (grande, arriba a la izquierda)
    st.sidebar.markdown(
        f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;justify-content:flex-start;overflow:visible">
  <img src="data:image/svg+xml;base64,{_LOGO_B64}" width="76" height="95" style="flex-shrink:0;object-fit:contain;display:block;vertical-align:middle" alt="CentineLA logo">
  <div style="font-size:38px;font-weight:500;letter-spacing:-1px;color:#3498db;line-height:1;white-space:nowrap;flex-shrink:1;min-width:0">
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
    st.sidebar.caption(f"Estado: {estado_actual}")
    st.sidebar.caption("CentineLA — Bairon Nicolas Calle Rivera · ITM · Territorio INN 2026")

    # Botón "Actualizar ahora" en sidebar
    st.sidebar.divider()
    if st.sidebar.button("🔄 Actualizar ahora", use_container_width=True):
        with st.spinner("Consultando IDEAM y recalculando..."):
            resultado = actualizar_ahora()
        if resultado["ok"]:
            st.sidebar.success(
                f"Listo — última lectura: {resultado['ultima_lectura']}"
            )
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(f"No se pudo actualizar: {resultado['error']}")

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
        "CentineLA · Bairon Nicolas Calle Rivera · ITM · Territorio INN 2026 · "
        "Datos: IDEAM (datos.gov.co) · Prototipo con gateway simulado"
    )


if __name__ == "__main__":
    main()
