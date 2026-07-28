"""
Fase 1, paso 2 — Pull histórico de precipitación para la(s) estación(es)
elegida(s) del CSV que generó 01_buscar_estaciones.py.

Uso:
    python 02_pull_historico.py <codigo_estacion> [codigo_estacion_2 ...]

Ej:
    python 02_pull_historico.py 27010060

Qué hace:
1. Inspecciona el esquema real de s54a-sgyg (precipitación histórica) y de
   ksew-j3zj (cuasi tiempo real) — no asumimos nombres de columna a ciegas,
   porque Socrata a veces los cambia entre datasets.
2. Intenta filtrar por el código de estación contra los campos más comunes
   (codigoestacion, codigo_estacion, estacion, codigo).
3. Guarda lo que encuentre en ../data/raw/historico_<codigo>.csv

Si el filtro automático no pega con ningún campo, el script imprime las
columnas reales para que ajustes el nombre a mano — mejor eso que asumir
mal y traer un CSV vacío sin que te enteres.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sodapy import Socrata

load_dotenv()

DATASET_HISTORICO = "s54a-sgyg"   # Precipitación histórica
DATASET_TIEMPO_REAL = "ksew-j3zj"  # Precipitación cuasi tiempo real
DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CAMPOS_CODIGO_POSIBLES = [
    "codigoestacion", "codigo_estacion", "codigoestacacion",
    "estacion", "codigo", "codigoestac",
]


def normaliza_texto(valor):
    if valor is None:
        return ""
    return str(valor).strip().upper()


def inspecciona(client, dataset_id, n=3):
    muestra = client.get(dataset_id, limit=n)
    if not muestra:
        print(f"[{dataset_id}] sin datos de muestra — dataset vacío o id incorrecto.")
        return []
    print(f"\n[{dataset_id}] columnas reales: {list(muestra[0].keys())}")
    return list(muestra[0].keys())


def detecta_campo_codigo(columnas):
    for c in CAMPOS_CODIGO_POSIBLES:
        if c in columnas:
            return c
    return None


def obtener_meta_catalogo(client, codigo_estacion):
    try:
        resultados = client.get("hp9r-jxuu", where=f"codigo='{codigo_estacion}'", limit=1)
    except Exception as e:
        print(f"No pude consultar el catálogo para {codigo_estacion}: {e}")
        return None

    if not resultados:
        return None

    return resultados[0]


def candidatos_codigo(codigo_estacion):
    codigo = str(codigo_estacion).strip()
    candidatos = []

    for valor in [codigo, codigo.lstrip("0"), codigo.zfill(10)]:
        if valor and valor not in candidatos:
            candidatos.append(valor)

    return candidatos


def fragmentos_nombre(nombre_estacion):
    if not nombre_estacion:
        return []

    nombre = normaliza_texto(nombre_estacion)
    nombre = nombre.split("[")[0].strip()
    partes = [p.strip() for p in nombre.split("-")]

    fragmentos = []
    for parte in partes:
        if len(parte) >= 5 and parte not in fragmentos:
            fragmentos.append(parte)

    return fragmentos


def pull_dataset(
    client,
    dataset_id,
    codigo_estacion,
    campo_codigo,
    meta_estacion=None,
    limit=500000,
    registrar_gaps=False,
):
    page_size = 5000
    intentos = []
    rangos_fallidos = []

    probe = client.get(dataset_id, limit=1)
    if not probe:
        if registrar_gaps:
            return pd.DataFrame(), rangos_fallidos
        return pd.DataFrame()

    columnas_probe = list(probe[0].keys())
    campo_fecha = "fechaobservacion"
    if campo_fecha not in columnas_probe:
        for candidato_fecha in ("fecha_observacion", "fecha", "datetime", "timestamp"):
            if candidato_fecha in columnas_probe:
                campo_fecha = candidato_fecha
                break

    if campo_fecha not in columnas_probe:
        print(
            f"  ADVERTENCIA: no pude detectar un campo de fecha usable en {dataset_id}. "
            f"Columnas vistas: {columnas_probe}"
        )
        if registrar_gaps:
            return pd.DataFrame(), rangos_fallidos
        return pd.DataFrame()

    def pagina_consulta(where_base):
        paginas = []
        pagina = 0
        total_filas = 0
        offset = 0

        while True:
            resultados = client.get(
                dataset_id,
                limit=page_size,
                offset=offset,
                order=":id",
                where=where_base,
            )

            pagina += 1
            filas_pagina = len(resultados)
            total_filas += filas_pagina

            if filas_pagina == 0:
                break

            paginas.append(pd.DataFrame.from_records(resultados))

            if filas_pagina < page_size:
                break

            offset += page_size

        df_final = pd.concat(paginas, ignore_index=True) if paginas else pd.DataFrame()
        return df_final

    def formatea_rango(inicio_dt, fin_dt):
        return f"{inicio_dt.date()} a {fin_dt.date()}"

    def descarga_rango_recursiva(where_base, inicio_dt, fin_dt, piso_dias=7):
        rango_txt = formatea_rango(inicio_dt, fin_dt)
        where_rango = (
            f"({where_base}) and {campo_fecha} between '{inicio_dt:%Y-%m-%dT%H:%M:%S}' "
            f"and '{fin_dt:%Y-%m-%dT%H:%M:%S}'"
        )
        print(f"  Rango {rango_txt}: consultando ...")
        try:
            df_rango = pagina_consulta(where_rango)
            print(f"  Rango {rango_txt}: {len(df_rango)} filas")
            return df_rango
        except Exception as e:
            if fin_dt - inicio_dt <= timedelta(days=piso_dias):
                print(f"  ADVERTENCIA: falló el rango {rango_txt} ({e}). Piso mínimo alcanzado, se registra como gap.")
                rangos_fallidos.append(rango_txt)
                return pd.DataFrame()

            print(f"  ADVERTENCIA: falló el rango {rango_txt} ({e}). Biseccionando...")

        mitad = inicio_dt + (fin_dt - inicio_dt) / 2
        if mitad <= inicio_dt:
            mitad = inicio_dt + timedelta(seconds=1)
        if mitad >= fin_dt:
            mitad = fin_dt - timedelta(seconds=1)

        if mitad <= inicio_dt or mitad >= fin_dt:
            print(f"  ADVERTENCIA: no pude bisecar más el rango {rango_txt}; se registra como gap.")
            rangos_fallidos.append(rango_txt)
            return pd.DataFrame()

        frames = []
        df_izq = descarga_rango_recursiva(where_base, inicio_dt, mitad, piso_dias=piso_dias)
        if not df_izq.empty:
            frames.append(df_izq)

        inicio_der = mitad
        if inicio_der <= fin_dt:
            df_der = descarga_rango_recursiva(where_base, inicio_der, fin_dt, piso_dias=piso_dias)
            if not df_der.empty:
                frames.append(df_der)

        df_rango = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print(f"  Rango {rango_txt}: {len(df_rango)} filas (vía bisección)")
        return df_rango

    def descarga_completa(where_base):
        frames = []
        for anio in range(2016, datetime.now().year + 1):
            inicio_anio = datetime(anio, 1, 1, 0, 0, 0)
            fin_anio = datetime(anio, 12, 31, 23, 59, 59)
            print(f"  Año {anio}: consultando ...")
            df_anio = descarga_rango_recursiva(where_base, inicio_anio, fin_anio)
            print(f"  Año {anio}: {len(df_anio)} filas")
            if not df_anio.empty:
                frames.append(df_anio)

        df_final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df_final.empty and campo_fecha in df_final.columns:
            df_final = df_final.sort_values(by=campo_fecha, kind="mergesort").reset_index(drop=True)
        return df_final

    for candidato in candidatos_codigo(codigo_estacion):
        intentos.append(f"{campo_codigo}='{candidato}'")

    if meta_estacion:
        nombre = meta_estacion.get("nombre") or meta_estacion.get("nombreestacion")
        departamento = normaliza_texto(meta_estacion.get("departamento"))
        municipio = normaliza_texto(meta_estacion.get("municipio"))
        for fragmento in fragmentos_nombre(nombre):
            clausula = f"{campo_codigo}='{codigo_estacion}' and nombreestacion like '%{fragmento}%'"
            if departamento:
                clausula += f" and departamento='{departamento}'"
            if municipio:
                clausula += f" and municipio='{municipio}'"
            if clausula not in intentos:
                intentos.append(clausula)

    for where in intentos:
        print(f"Consultando [{dataset_id}] where {where} ...")
        try:
            probe = client.get(dataset_id, where=where, limit=1)
        except Exception as e:
            print(f"  Falló esta consulta ({e}).")
            continue

        if not probe:
            continue

        print(f"  Coincidencia con: {where}")
        try:
            df = descarga_completa(where)
            print(f"  Paginación por años completada: {len(df)} filas")
            if registrar_gaps:
                return df, rangos_fallidos
            return df
        except Exception as e:
            print(f"  ADVERTENCIA: falló la descarga por años ({e}).")
            raise

    print(
        f"  ADVERTENCIA: no se encontró una coincidencia segura para {codigo_estacion} "
        f"en {dataset_id}."
    )
    if registrar_gaps:
        return pd.DataFrame(), rangos_fallidos
    return pd.DataFrame()


def main():
    if len(sys.argv) < 2:
        print("Uso: python 02_pull_historico.py <codigo_estacion> [otro_codigo ...]")
        sys.exit(1)

    codigos = sys.argv[1:]
    _partes_dominio = [chr(119)*2+chr(119), "datos", "gov", "co"]

    username = os.environ.get("SODAPY_USERNAME")
    password = os.environ.get("SODAPY_PASSWORD")
    if not username or not password:
        raise ValueError(
            "SODAPY_USERNAME y SODAPY_PASSWORD deben estar definidos en el entorno. "
            "Copia .env.example a .env y completa los valores."
        )

    client = Socrata(
        ".".join(_partes_dominio),
        None,
        username=username,
        password=password,
        timeout=180,
    )

    cols_hist = inspecciona(client, DATASET_HISTORICO)
    cols_rt = inspecciona(client, DATASET_TIEMPO_REAL)

    campo_hist = detecta_campo_codigo(cols_hist)
    campo_rt = detecta_campo_codigo(cols_rt)

    if campo_hist is None:
        print(
            f"\nNo detecté automáticamente el campo de código de estación en "
            f"{DATASET_HISTORICO}. Mirá la lista de columnas arriba y ajustá "
            f"CAMPOS_CODIGO_POSIBLES en el script."
        )

    for codigo in codigos:
        print(f"\n=== Estación {codigo} ===")

        meta = obtener_meta_catalogo(client, codigo)
        if meta:
            print(f"  Catálogo: {meta.get('nombre')} | {meta.get('municipio')} | {meta.get('estado')}")
        else:
            print("  Catálogo: no encontré metadata para este código.")

        if campo_hist:
            df_hist, rangos_fallidos = pull_dataset(
                client,
                DATASET_HISTORICO,
                codigo,
                campo_hist,
                meta_estacion=meta,
                registrar_gaps=True,
            )
            print(f"  Histórico: {len(df_hist)} filas")
            if not df_hist.empty:
                dedup_cols = [campo_hist, "fechaobservacion", "codigosensor", "valorobservado"]
                if all(col in df_hist.columns for col in dedup_cols):
                    df_hist = df_hist.drop_duplicates(subset=dedup_cols)
                else:
                    print(f"  Aviso: no pude aplicar dedup por columnas {dedup_cols} porque faltan columnas en el resultado.")

                if "fechaobservacion" in df_hist.columns:
                    df_hist = df_hist.sort_values(by="fechaobservacion", kind="mergesort").reset_index(drop=True)

                if (
                    "codigosensor" in df_hist.columns
                    and "descripcionsensor" in df_hist.columns
                    and df_hist["codigosensor"].nunique(dropna=False) > 1
                ):
                    print("  Resumen de sensores:")
                    print(df_hist.groupby(["codigosensor", "descripcionsensor"]).size())

                DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
                out = DATA_RAW_DIR / f"historico_{codigo}.csv"
                df_hist.to_csv(out, index=False)
                print(f"  Guardado: {out}")

            else:
                print(
                    "  Sin filas históricas para esta estación en s54a-sgyg. "
                    "No se guarda CSV vacío; revisá si existe histórico real para este código."
                )

        if campo_rt:
            df_rt = pull_dataset(client, DATASET_TIEMPO_REAL, codigo, campo_rt, meta_estacion=meta)
            print(f"  Tiempo real: {len(df_rt)} filas")
            if not df_rt.empty:
                DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
                out = DATA_RAW_DIR / f"tiemporeal_{codigo}.csv"
                df_rt.to_csv(out, index=False)
                print(f"  Guardado: {out}")
            else:
                print(
                    "  Sin filas de tiempo real para esta estación. "
                    "Si el nodo existe en catálogo, puede que no publique a este dataset."
                )

        if 'rangos_fallidos' in locals() and rangos_fallidos:
            print("\n⚠️  ADVERTENCIA — DATOS INCOMPLETOS para esta estación:")
            print("Los siguientes rangos NO se pudieron descargar tras reintentos:")
            for rango in rangos_fallidos:
                print(f"  - {rango}")
            print("El CSV guardado NO cubre estos períodos. Revisar manualmente o reintentar más tarde.")

            DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
            gaps_out = DATA_RAW_DIR / f"GAPS_{codigo}.txt"
            with open(gaps_out, "w", encoding="utf-8") as f:
                for rango in rangos_fallidos:
                    f.write(f"- {rango}\n")


if __name__ == "__main__":
    main()
