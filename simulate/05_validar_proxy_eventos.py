"""
Fase 2, paso 4 - Validacion de eventos del proxy SCS-CN.

Carga data/processed/proxy_q_la_honda.csv, revisa ventanas de ±48h alrededor
de eventos conocidos, calcula percentiles de Q_scs_proxy y guarda una grafica
con la serie completa y marcas verticales.

Uso:
    python 05_validar_proxy_eventos.py

Salida:
    ../data/processed/proxy_q_la_honda_eventos.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROXY_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda.csv"
PLOT_PATH = DATA_PROCESSED_DIR / "proxy_q_la_honda_eventos.png"

# Fechas exactas pendientes — ver PQRSD a atencionusuario@metropol.gov.co o fuente DAGRD directa.
# NO inferir de la serie propia.
EVENTOS_CONOCIDOS = {}


def cargar_proxy():
    if not PROXY_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo esperado: {PROXY_PATH}")

    df = pd.read_csv(PROXY_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ventana_evento(df, fecha_evento):
    inicio = fecha_evento - pd.Timedelta(hours=48)
    fin = fecha_evento + pd.Timedelta(hours=48)
    ventana = df[(df["timestamp"] >= inicio) & (df["timestamp"] <= fin)].copy()
    return inicio, fin, ventana


def percentil_de_valor(serie, valor):
    serie_no_na = serie.dropna()
    if serie_no_na.empty or pd.isna(valor):
        return float("nan")
    return float((serie_no_na <= valor).mean() * 100)


def imprimir_evento(nombre_evento, fecha_evento, df):
    inicio, fin, ventana = ventana_evento(df, fecha_evento)
    if ventana.empty:
        print(f"\n=== {nombre_evento} ===")
        print(f"Fecha base: {fecha_evento}")
        print("No hay datos en la ventana de ±48h.")
        return None

    target_row = ventana.loc[ventana["timestamp"].sub(fecha_evento).abs().idxmin()]
    q_evento = float(target_row["Q_scs_proxy"]) if pd.notna(target_row["Q_scs_proxy"]) else float("nan")
    p_evento = float(target_row["P_basin"]) if pd.notna(target_row["P_basin"]) else float("nan")
    p_acc_evento = float(target_row["P_acc_24h"]) if pd.notna(target_row["P_acc_24h"]) else float("nan")
    pct = percentil_de_valor(df["Q_scs_proxy"], q_evento)
    top5_threshold = float(df["Q_scs_proxy"].dropna().quantile(0.95))
    bandera_roja = pd.isna(q_evento) or q_evento < top5_threshold

    print(f"\n=== {nombre_evento} ===")
    print(f"Fecha base: {fecha_evento}")
    print(f"Ventana: {inicio} -> {fin}")
    print(
        "Valor en la fecha base mas cercana disponible: "
        f"timestamp={target_row['timestamp']}, P_basin={p_evento}, P_acc_24h={p_acc_evento}, Q_scs_proxy={q_evento}"
    )
    print(f"Percentil de Q_scs_proxy respecto a toda la serie: {pct:.2f}")
    print(f"Umbral top 5% de toda la serie: {top5_threshold:.6f}")

    if bandera_roja:
        print("BANDERA ROJA: el evento NO cae en el top 5% de Q_scs_proxy de la serie completa.")
    else:
        print("OK: el evento si cae en el top 5% de Q_scs_proxy de la serie completa.")

    columnas = ["timestamp", "P_basin", "P_acc_24h", "Q_scs_proxy", "n_estaciones_disponibles"]
    print("\nVentana completa de ±48h:")
    print(ventana[columnas].to_string(index=False))

    return {
        "nombre": nombre_evento,
        "fecha_evento": fecha_evento,
        "q_evento": q_evento,
        "pct": pct,
        "top5_threshold": top5_threshold,
        "bandera_roja": bandera_roja,
    }


def graficar(df, eventos):
    plt.figure(figsize=(16, 7))
    plt.plot(df["timestamp"], df["Q_scs_proxy"], linewidth=0.9, color="#1f4e79", label="Q_scs_proxy")

    colores = ["#b22222", "#7a3e00"]
    for idx, (nombre, fecha_evento) in enumerate(eventos.items()):
        plt.axvline(fecha_evento, color=colores[idx % len(colores)], linestyle="--", linewidth=2.0, label=nombre)

    plt.title("CentineLA - Proxy SCS-CN y eventos de validacion")
    plt.xlabel("timestamp")
    plt.ylabel("Q_scs_proxy")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)
    plt.close()
    print(f"\nGrafica guardada en: {PLOT_PATH}")


def main():
    df = cargar_proxy()
    print(f"Archivo cargado: {PROXY_PATH}")
    print(f"len(df): {len(df)}")
    print(f"rango fechas: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    resultados = []
    if not EVENTOS_CONOCIDOS:
        print(
            "\nADVERTENCIA: EVENTOS_CONOCIDOS esta vacio. Se omite la validacion de eventos "
            "para no inferir ground truth desde la serie propia."
        )
    else:
        for nombre, fecha_evento in EVENTOS_CONOCIDOS.items():
            resultados.append(imprimir_evento(nombre, fecha_evento, df))

    graficar(df, EVENTOS_CONOCIDOS)

    print("\n=== Resumen ===")
    for resultado in resultados:
        if resultado is None:
            continue
        estado = "BANDERA ROJA" if resultado["bandera_roja"] else "OK"
        print(
            f"{resultado['nombre']}: {estado} | Q={resultado['q_evento']:.6f} | "
            f"percentil={resultado['pct']:.2f} | umbral_top5={resultado['top5_threshold']:.6f}"
        )


if __name__ == "__main__":
    main()