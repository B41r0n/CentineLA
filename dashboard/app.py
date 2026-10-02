"""
CentineLA — Dashboard Streamlit
Vista Pública + Vista Operador (JAC)

Uso:
    streamlit run dashboard/app.py
"""

import io
import math
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

# ── métricas operativas (calculadas en 08_clasificador_6h.py) ──
RECALL_OPERATIVO = 0.743
PRECISION_OPERATIVA = 0.400
UMBRAL_PRECAUCION = 0.30
UMBRAL_ALERTA = 0.70


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


def semaforo_html(estado: str) -> str:
    if estado == "ALERTA":
        bg, emoji, texto = "#e74c3c", "🔴", "ALERTA — PRÓXIMAS 6 HORAS"
    elif estado == "PRECAUCIÓN":
        bg, emoji, texto = "#f39c12", "🟡", "PRECAUCIÓN — CONDICIONES A REVISAR"
    else:
        bg, emoji, texto = "#2ecc71", "🟢", "NORMAL"
    return f"""
    <div style="
        background-color:{bg};
        border-radius:16px;
        padding:24px 32px;
        text-align:center;
        color:white;
        font-size:2rem;
        font-weight:bold;
        letter-spacing:0.05em;
        box-shadow:0 4px 12px rgba(0,0,0,0.3);
    ">{emoji}&nbsp;&nbsp;{texto}</div>
    """


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
    # Crear el mapa con location+zoom_start como fallback, pero usar fit_bounds
    # para forzar el encuadre en las 3 anclas independientemente del height.
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=13, tiles="OpenStreetMap")
    pad = 0.015   # ~1.5 km de margen alrededor de las anclas
    m.fit_bounds(
        [[min(lats_a) - pad, min(lons_a) - pad],
         [max(lats_a) + pad, max(lons_a) + pad]]
    )

    coords = [[a["lat"], a["lon"]] for a in anclas]
    folium.PolyLine(coords, color="#1f77b4", weight=3, opacity=0.7).add_to(m)

    if estado == "ALERTA":
        color_nodo = "red"
    elif estado == "PRECAUCIÓN":
        color_nodo = "orange"
    else:
        color_nodo = "green"
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
        "Mapa interactivo OpenStreetMap. "
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
    if estado == "ALERTA":
        color = "#e74c3c"
    elif estado == "PRECAUCIÓN":
        color = "#f39c12"
    else:
        color = "#2ecc71"

    fig, ax = plt.subplots(figsize=(9, 6))

    # 1) Recorrido real — linea por las 3 anclas (siempre pasa por los 3 puntos).
    ax.plot(lons_linea, lats_linea, color="#1f77b4", linewidth=2.5, zorder=2)

    # 2) Marcadores de ancla: circulo HUECO centrado exactamente en el vertice.
    #    Se usa marker='o' (circulo) porque su centro geometrico == centro visual,
    #    sin ambiguedad de orientacion (marker='^' tenia el centro de su bounding-box
    #    en la coordenada, haciendo que el apice del triangulo quedara visualmente
    #    por encima del vertice de la linea, aunque las coords fuesen identicas).
    lat_min = min(a["lat"] for a in anclas)
    lat_max_a = max(a["lat"] for a in anclas)
    lon_min_a = min(a["lon"] for a in anclas)
    lon_max_a = max(a["lon"] for a in anclas)
    lat_rng = (lat_max_a - lat_min) or 1.0
    lon_rng = (lon_max_a - lon_min_a) or 1.0

    for a in anclas:
        ax.scatter(
            [a["lon"]], [a["lat"]],
            s=200, facecolors="white", edgecolors="#1f77b4",
            linewidths=2.5, zorder=5, marker="o",
        )
        # Dirección del label: hacia arriba si el ancla está cerca del borde inferior
        # (evita recorte); hacia la izquierda si está cerca del borde derecho.
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
            fontsize=7, color="#555555", style="italic",
        )

    # 3) Nodos de monitoreo — circulos coloreados por estado, SIN label de texto.
    #    Los nodos son posiciones ilustrativas proporcionales; sus nombres
    #    ("Nodo 1", "Nodo 2"...) no aportan informacion geografica real y
    #    generan solapamiento cuando un nodo cae sobre una ancla.
    ax.scatter(lons_nodos, lats_nodos, c=color, s=110, zorder=4, edgecolors="white", linewidths=1.5)

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(
        "CentineLA — Q. La Honda (posiciones ilustrativas, no GPS de campo)",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    # Leyenda en lower right (area vacia en este plot — los datos se concentran
    # en las esquinas left y upper right, dejando lower right libre).
    handle_estado = mpatches.Patch(color=color, label=estado)
    handle_ancla = mlines.Line2D(
        [], [], marker="o", color="w",
        markerfacecolor="white", markeredgecolor="#1f77b4",
        markersize=10, markeredgewidth=2.5, label="Ancla geográfica",
    )
    ax.legend(handles=[handle_estado, handle_ancla], loc="lower right", fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
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

def _tarjeta_html(icono: str, label: str, valor: str, unidad: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div style="font-size:1.6rem;margin-bottom:4px">{icono}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{valor}</div>'
        f'<div style="font-size:11px;color:#888;margin-top:2px">{unidad}</div>'
        f'</div>'
    )


def _color_proba(proba: float) -> str:
    if proba >= UMBRAL_ALERTA:
        return "#e74c3c"
    if proba >= UMBRAL_PRECAUCION:
        return "#f39c12"
    return "#2ecc71"


def _estado_semaforo_html(estado: str, proba: float) -> str:
    if estado == "ALERTA":
        bg, emoji, texto = "#e74c3c", "🔴", "ALERTA — PRÓXIMAS 6 HORAS"
    elif estado == "PRECAUCIÓN":
        bg, emoji, texto = "#f39c12", "🟡", "PRECAUCIÓN — CONDICIONES A REVISAR"
    else:
        bg, emoji, texto = "#2ecc71", "🟢", "NORMAL"
    color_p = _color_proba(proba)
    return f"""
    <div style="display:grid;grid-template-columns:60% 38%;gap:2%;margin-bottom:8px">
      <div style="background:{bg};border-radius:14px;padding:28px 20px;
                  text-align:center;color:white;font-size:clamp(18px,3vw,26px);
                  font-weight:700;letter-spacing:.04em;
                  box-shadow:0 4px 14px rgba(0,0,0,.25)">
        {emoji}&nbsp;{texto}
      </div>
      <div class="metric-card" style="text-align:center;display:flex;
                  flex-direction:column;justify-content:center;align-items:center">
        <div class="metric-label">PROBABILIDAD</div>
        <div class="metric-value" style="color:{color_p}">{proba:.1%}</div>
        <div style="font-size:11px;color:#888">umbrales: PRECAUCIÓN ≥{UMBRAL_PRECAUCION} | ALERTA ≥{UMBRAL_ALERTA}</div>
      </div>
    </div>
    """


def _calendario_html(log: pd.DataFrame, n_dias: int = 90) -> str:
    ultima_fecha = log["timestamp"].dt.date.max()
    inicio = ultima_fecha - pd.Timedelta(days=n_dias - 1)
    rango = pd.date_range(inicio, ultima_fecha, freq="D")

    # Estado máximo por día: 0=NORMAL, 1=PRECAUCIÓN, 2=ALERTA
    estado_map = {"NORMAL": 0, "PRECAUCIÓN": 1, "ALERTA": 2}
    colores = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
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
            + ';border-radius:2px;aspect-ratio:1;min-height:14px"></div>'
        )
        celdas.append(celda)

    grid_html = (
        '<div style="display:grid;grid-template-columns:repeat(18,1fr);'
        'gap:3px;max-width:100%;margin-bottom:6px">'
        + "".join(celdas)
        + "</div>"
    )
    cuadro_n = (
        '<span style="display:inline-block;width:12px;height:12px;'
        'background:#2ecc71;border-radius:2px;vertical-align:middle"></span>'
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
        '<div style="display:flex;gap:16px;font-size:12px;color:#555">'
        + cuadro_n + " Normal&nbsp;&nbsp;"
        + cuadro_p + " Precaución&nbsp;&nbsp;"
        + cuadro_a + " Alerta"
        + "</div>"
    )
    return grid_html + leyenda


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
    st.header("Q. La Honda — Estado de alerta en tiempo real")

    ultima = log.iloc[-1]
    estado_actual = str(ultima["estado"])
    proba_actual = float(ultima["proba_alerta"])

    # ── PASO 2: semáforo + probabilidad ────────────────────────────────────
    st.markdown(
        _estado_semaforo_html(estado_actual, proba_actual),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Última lectura disponible: **{ultima['timestamp']}**"
    )
    st.divider()

    # ── PASO 3: tarjetas de lluvia acumulada ───────────────────────────────
    st.subheader("Lluvia acumulada")
    ult_24h, ult_30d, anio_mm, ref_fecha = _lluvia_acumulada(proxy)
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:8px">'
        + _tarjeta_html("🌧️", "Últ. 24 h", f"{ult_24h:.1f}", "mm")
        + _tarjeta_html("📅", "Últ. 30 días", f"{ult_30d:.1f}", "mm")
        + _tarjeta_html("📆", "Año en curso", f"{anio_mm:.1f}", "mm")
        + "</div>"
        + '<p style="font-size:11px;color:#888;margin-top:0">'
        + "Ref. histórica: " + ref_fecha
        + ". Ventanas relativas al último dato disponible — no al reloj del sistema.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── PASO 4: mapa más grande ────────────────────────────────────────────
    st.subheader("Mapa de nodos")
    nodos = interpolar_nodos(ANCLAS, N_NODOS)
    mostrar_mapa(ANCLAS, nodos, estado_actual)
    st.divider()

    # ── PASO 5: calendario de alertas (últimos 90 días del log) ───────────
    st.subheader("Calendario de alertas — últimos 90 días")
    st.markdown(
        _calendario_html(log, n_dias=90),
        unsafe_allow_html=True,
    )
    st.caption(
        "Cada cuadro = un día. Verde = NORMAL, amarillo = al menos una hora PRECAUCIÓN, "
        "rojo = al menos una hora ALERTA."
    )


# ── helpers de Vista Operador ────────────────────────────────────────────────

def _grafico_historico(log: pd.DataFrame, proxy: pd.DataFrame) -> None:
    st.subheader("Historial de lluvia y escorrentía estimada")

    fecha_min = proxy["timestamp"].dt.date.min()
    fecha_max = proxy["timestamp"].dt.date.max()
    valor_inicial = (fecha_max - pd.Timedelta(days=90), fecha_max)

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
        name="P_basin (mm/h)", line=dict(color="#2980b9", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=proxy_rango["timestamp"], y=proxy_rango["Q_scs_proxy"],
        name="Q_scs_proxy (mm)", line=dict(color="#8e44ad", width=1), yaxis="y2",
    ))
    if not precauciones.empty:
        fig.add_trace(go.Scatter(
            x=precauciones["timestamp"], y=precauciones["proba_alerta"],
            mode="markers", name="PRECAUCIÓN",
            marker=dict(color="#f39c12", size=6, symbol="x"), yaxis="y2",
        ))
    if not alertas.empty:
        fig.add_trace(go.Scatter(
            x=alertas["timestamp"], y=alertas["proba_alerta"],
            mode="markers", name="ALERTA",
            marker=dict(color="#e74c3c", size=6, symbol="x"), yaxis="y2",
        ))
        for ts in alertas["timestamp"].iloc[::max(1, len(alertas) // 200)]:
            fig.add_vline(x=ts, line=dict(color="red", width=0.6, dash="dot"), opacity=0.3)

    fig.update_layout(
        title="P_basin (lluvia) y Q_scs_proxy (escorrentía) — marcadores = PRECAUCIÓN/ALERTA",
        xaxis=dict(title="Fecha"),
        yaxis=dict(title=dict(text="P_basin (mm/h)", font=dict(color="#2980b9"))),
        yaxis2=dict(
            title=dict(text="Q_scs_proxy / probabilidad", font=dict(color="#8e44ad")),
            overlaying="y", side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Alertas en el período seleccionado: **{len(alertas)}**. "
        "Las líneas verticales rojas se muestran submuestreadas para rendimiento."
    )


# ── vista operador ────────────────────────────────────────────────────────────

def vista_operador(log: pd.DataFrame, proxy: pd.DataFrame, modelo) -> None:
    st.header("Vista Operador (JAC) — Diagnóstico técnico")

    # log reciente
    st.subheader("Log de gateway — últimas 100 filas")
    st.dataframe(log.tail(100), use_container_width=True)
    st.divider()

    # gráfico técnico (movido desde Vista Pública)
    _grafico_historico(log, proxy)
    st.divider()

    # métricas del modelo
    st.subheader("Métricas del modelo (umbrales ternarios: PRECAUCIÓN ≥0.30, ALERTA ≥0.70)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Recall clase positiva", f"{RECALL_OPERATIVO:.3f}",
        help="De los eventos reales ¿cuántos detecta? (split test cronológico 2025-07-06 → 2026-07-09)"
    )
    c2.metric(
        "Precision clase positiva", f"{PRECISION_OPERATIVA:.3f}",
        help="De las alertas disparadas ¿cuántas son eventos reales?"
    )
    c3.metric(
        "Umbrales de decisión", f"{UMBRAL_PRECAUCION} / {UMBRAL_ALERTA}",
        help="PRECAUCIÓN ≥ 0.30 | ALERTA ≥ 0.70"
    )
    st.caption(
        "Métricas calculadas sobre split cronológico 82/18. "
        "No se recalculan en vivo — fuente: `08_clasificador_6h.py`."
    )
    st.divider()

    # feature importances
    st.subheader("Importancia de variables — clf_6h.joblib")
    try:
        importancias = dict(zip(FEATURE_COLS, modelo.feature_importances_))
        df_imp = (
            pd.DataFrame(
                {"feature": list(importancias.keys()), "importance": list(importancias.values())}
            )
            .sort_values("importance", ascending=True)
        )
        fig_imp = go.Figure(go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            marker_color="#2980b9",
        ))
        fig_imp.update_layout(
            height=500,
            title="Feature importance (RandomForestClassifier, class_weight=balanced)",
            margin=dict(l=160),
            xaxis_title="Importancia",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as exc:
        st.warning(f"No se pudo graficar importancias: {exc}")

    st.divider()

    # inspección por fecha
    st.subheader("Inspección de punto histórico")
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
    hora_sel = st.selectbox("Hora (UTC-5 local)", opciones_hora, key="hora_inspeccion")

    fila = log_dia[log_dia["timestamp"].dt.strftime("%H:%M") == hora_sel].iloc[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Timestamp:** `{fila['timestamp']}`")
        st.markdown(f"**Estado:** `{fila['estado']}`")
        st.markdown(f"**Probabilidad:** `{fila['proba_alerta']:.4f}`")

    with col_b:
        cols_mostrar = [c for c in ["P_basin", "Q_actual", "Q_lag_1h", "roll_sum_24h", "hora_dia", "mes"]
                        if c in fila.index]
        if cols_mostrar:
            st.dataframe(
                pd.DataFrame([{c: round(float(fila[c]), 6) for c in cols_mostrar}])
                .T.rename(columns={0: "valor"}),
                use_container_width=True,
            )

    st.divider()
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
    st.subheader("Probar el modelo — simulador interactivo")
    st.caption(
        "Introduce valores hipotéticos de lluvia y caudal para ver cómo responde "
        "el clasificador en tiempo real. Los campos derivados (lag_12h, Q_lags, etc.) "
        "se aproximan automáticamente a partir de los sliders."
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
        lag_1h  = st.slider("Lluvia última hora  (lag_1h) mm",  0.0, 50.0, step=0.5, key="sim_lag_1h")
        lag_3h  = st.slider("Lluvia últimas 3h   (lag_3h) mm",  0.0, 50.0, step=0.5, key="sim_lag_3h")
        lag_6h  = st.slider("Lluvia últimas 6h   (lag_6h) mm",  0.0, 50.0, step=0.5, key="sim_lag_6h")
        q_act   = st.number_input("Q actual (proxy nivel, mm)", min_value=0.0, max_value=20.0, step=0.1, key="sim_Q")
    with col2:
        roll12  = st.slider("Acum. 12h (roll_sum_12h) mm", 0.0, 50.0, step=0.5, key="sim_roll12")
        roll24  = st.slider("Acum. 24h (roll_sum_24h) mm", 0.0, 50.0, step=0.5, key="sim_roll24")
        roll48  = st.slider("Acum. 48h (roll_sum_48h) mm", 0.0, 50.0, step=0.5, key="sim_roll48")

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

    if st.button("Calcular predicción", type="primary"):
        x_df = pd.DataFrame([x_vec])[FEATURE_COLS]
        proba = float(modelo.predict_proba(x_df)[0, 1])
        estado_pred = clasificar_estado(proba)
        st.markdown(
            _estado_semaforo_html(estado_pred, proba),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Probabilidad bruta: **{proba:.4f}** — umbrales: PRECAUCIÓN ≥{UMBRAL_PRECAUCION}, ALERTA ≥{UMBRAL_ALERTA}. "
            "Las features derivadas son aproximaciones; en producción se calculan de la serie real."
        )


# ── app principal ─────────────────────────────────────────────────────────────

CSS_GLOBAL = """
<style>
.metric-card {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(0,0,0,.08);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.metric-value {
    font-size: clamp(28px, 5vw, 40px);
    font-weight: 600;
    line-height: 1.1;
    margin: 4px 0;
}
.metric-label {
    font-size: 13px;
    color: #777;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 2px;
}
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="CentineLA — Alerta Q. La Honda",
        page_icon="💧",
        layout="wide",
    )
    st.markdown(CSS_GLOBAL, unsafe_allow_html=True)
    st.title("💧 CentineLA — Alerta temprana de crecientes súbitas | Q. La Honda")

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

    vista = st.sidebar.radio(
        "Seleccionar vista",
        ["Vista Pública", "Vista Operador (JAC)"],
    )
    st.sidebar.caption(f"Filas en log: {len(log):,}  |  Último estado: {log.iloc[-1]['estado']}")
    st.sidebar.caption("CentineLA — Bairon Nicolas Calle Rivera · ITM · Territorio INN 2026")

    if vista == "Vista Pública":
        vista_publica(log, proxy)
    else:
        if modelo is None:
            st.error("clf_6h.joblib no disponible — Vista Operador no puede mostrar importancias.")
        else:
            vista_operador(log, proxy, modelo)


if __name__ == "__main__":
    main()
