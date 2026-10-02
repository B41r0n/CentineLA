# CentineLA

Sistema de alerta temprana de crecientes súbitas para la **Quebrada La Honda**, Comuna 4 Aranjuez, Medellín.

> Repositorio real ubicado en: `C:\Users\Nico\Desktop\CentineLA`

## Autor

Bairon Nicolas Calle Rivera — estudiante de Tecnología en Sistemas, ITM (Instituto Tecnológico Metropolitano), Medellín.
Proyecto individual — Territorio INN 2026, Reto #7.

---

## 1. Qué es y por qué existe

### Problema

La **Quebrada La Honda** es una microcuenca de ladera nororiental de Medellín que drena hacia el río Aburrá-Medellín a la altura de la Comuna 4 (Aranjuez). Tiene tres características que la hacen crítica:

* **Sin sistema de alerta temprana local (SAT):** no hay una red de sensores ni alarmas comunitarias dedicada al cauce.
* **Topografía vulnerable:** tramo alto (>30 % de pendiente, cabecera en el Parque Ecoturístico Arví), tramo medio (12-25 %, barrios La Honda, La Hondita, Versalles 1 y 2, El Raizal) y tramo bajo (6-12 %, Jardín Botánico, Moravia).
* **Eventos documentados:** avenidas torrenciales en **1996**, **2022** y **abril de 2026**, que afectaron el tramo medio y bajo.

El sistema **CentineLA** intenta cerrar esa brecha usando datos públicos de precipitación del IDEAM (`datos.gov.co`), un proxy hidrológico SCS-CN, modelos de Machine Learning y un dashboard de monitoreo.

### Qué resuelve

1. Consolida un proxy horario de escorrentía (`Q_scs_proxy`) a partir de estaciones pluviométricas cercanas.
2. Entrena un clasificador que, 6 h antes, estima el nivel de riesgo en tres estados: NORMAL, PRECAUCIÓN o ALERTA.
3. Simula cómo un gateway embebido (RPi4) calcularía esas features en tiempo real.
4. Expone el resultado en un dashboard de dos vistas: pública y operador/JAC.

---

## 2. Arquitectura general

```text
CUENCA ALTA                         TRAMO MEDIO                         TRAMO BAJO
(Parque Ecoturístico Arví)          (Las Granjas / La Honda)            (Parque Montecarlo / Moravia)

[Pluviómetro tipping bucket]        [Cámara IP + CV]                    [Cámara IP + CV]
 + LoRa TX                          + Sensor turbidez óptico            + Sensor turbidez óptico
                                    + Sensor ultrasónico JSN-SR04T      + Sensor ultrasónico
                                    + LoRa TX                           + LoRa TX
         │                                   │                                  │
         └───────────────────────────────────┴──────────────────────────────────┘
                                             │ LoRa 915 MHz (ISM Colombia)
                                  ┌──────────▼──────────┐
                                  │   GATEWAY           │
                                  │   Raspberry Pi 4    │
                                  │   + HAT LoRa RAK2287│
                                  │   + OpenCV          │
                                  │   + Modelo ML       │
                                  │   + Broker MQTT     │
                                  └──────────┬──────────┘
                ┌────────────────────────────┼────────────────────────────┐
                ▼ LoRa broadcast             ▼ Internet (4G backup)       │
     [Postes: LED semáforo 360° + sirena]   [WhatsApp / DAGRD si rojo]   [Feed JSON → SIATA]
                                            [Dashboard web]
```

### Componentes físicos y rol

| Componente | Rol en la arquitectura completa |
|------------|---------------------------------|
| **Nodo cuenca alta** | Pluviómetro + LoRa. Es la fuente más cercana a la cabecera; mide precipitación antes de que el pico llegue al tramo medio. |
| **Nodo tramo medio/bajo** | Cámara IP (marcas de referencia blanco/negro para OpenCV), sensor de turbidez óptico, sensor ultrasónico JSN-SR04T para nivel. Confirman el pico con múltiples fuentes. |
| **Gateway RPi4 + RAK2287** | Recibe LoRa, corre feature-engineering, modelo y decide estado. Publica alerta. |
| **Broker MQTT / WhatsApp** | Canal de alerta a la comunidad y a DAGRD. |
| **Dashboard Streamlit** | Vista pública + vista operador para JAC; consume el log del gateway y el modelo `clf_6h.joblib`. |

> En el repositorio actual, la capa física aún no está implementada. Lo que existe es la **línea de datos IDEAM → proxy → ML → gateway simulado → dashboard**.

---

## 3. Estructura del repositorio

```text
CentineLA/
├── .gitignore                              # Ignora venv, __pycache__, CSVs crudos/procesados y modelos .joblib
├── .vscode/settings.json                   # Configuración local de VS Code
├── requirements.txt                        # Dependencias raíz (ahora completas)
├── CENTINELA_CONTEXTO_TECNICO.md           # Contexto interno para asistentes de código
├── STATUS_REPORT.md                        # Auditoría de solo lectura del 2026-07-21
├── RESUMEN_SESION_CENTINELA_2026-07-11.md  # Resumen de la sesión que cerró proxy, validación y ETL
│
├── scripts/
│   ├── 01_buscar_estaciones.py             # Filtra catálogo IDEAM por Valle de Aburrá y rankea por distancia a La Honda
│   └── 02_pull_historico.py                # Descarga precipitación histórica/tiempo real por estación
│
├── data/
│   ├── raw/
│   │   ├── estaciones_candidatas.csv       # Salida de 01_buscar_estaciones.py (NO trackeado por .gitignore)
│   │   ├── historico_0027015290.csv        # Pajarito (NO trackeado)
│   │   ├── historico_0027015310.csv        # Metromedellín (NO trackeado)
│   │   └── historico_0027015330.csv        # Olaya Herrera (NO trackeado)
│   │
│   └── processed/                          # Trackeado para deploy
│       ├── proxy_q_la_honda.csv            # Proxy SCS-CN horario
│       ├── proxy_q_la_honda_eventos.png    # Serie completa + marcas de eventos (vacío hoy)
│       ├── filas_baja_calidad.csv          # Filas con n_estaciones_disponibles < 2
│       ├── dataset_6h.csv                  # Dataset de entrenamiento (horizonte 6 h)
│       ├── dataset_12h.csv                 # Dataset de entrenamiento (horizonte 12 h)
│       ├── dataset_24h.csv                 # Dataset de entrenamiento (horizonte 24 h)
│       └── log_gateway_simulado.csv        # Resultado de la simulación del gateway
│
├── simulate/
│   ├── cuenca_la_honda_params.py           # Área, longitud, tramos, CN compuesto (dato-duro vs supuesto)
│   ├── 04_scs_cn_proxy.py                  # Construye el proxy SCS-CN
│   ├── 05_validar_proxy_eventos.py         # Valida proxy contra eventos conocidos (modo exploratorio)
│   ├── 06_etl_features.py                  # ETL de features, target y split cronológico
│   ├── 07_entrenar_modelo_6h.py            # 3 enfoques de RandomForestRegressor
│   ├── 08_clasificador_6h.py               # RandomForestClassifier: NORMAL / PRECAUCIÓN / ALERTA (modelo en producción)
│   ├── 09_gateway_simulado.py              # Gateway simulado en modo streaming
│   └── models/                             # Trackeado para deploy
│       ├── clf_6h.joblib
│       └── clf_6h_baseline_20260913.joblib # Respaldo del modelo previo
│
└── dashboard/
    ├── app.py                              # Dashboard Streamlit
    └── _check_fixes.py                     # Script de verificación de fixes y simulador
```

---

## 4. Pipeline de datos

### 4.1 `scripts/01_buscar_estaciones.py` — Buscar estaciones IDEAM cercanas

**Qué hace paso a paso:**

1. Se conecta a `www.datos.gov.co` vía `sodapy.Socrata` sin `app_token`.
2. Descarga el catálogo nacional de estaciones (`hp9r-jxuu`) filtrando `Departamento="Antioquia"`, `limit=5000`.
3. Extrae lat/lon de columnas planas (`latitud`/`longitud`) o del objeto anidado `ubicaci_n`.
4. Filtra por municipios del Valle de Aburrá.
5. Calcula distancia Haversine desde un punto de referencia fijo.
6. Ordena por distancia y exporta `data/raw/estaciones_candidatas.csv`.

**Entrada:** ninguna (solo conexión a internet).

**Salida:** `data/raw/estaciones_candidatas.csv`.

**Constantes clave (valores reales del código):**

```python
LAT_LA_HONDA = 6.271
LON_LA_HONDA = -75.564

DATASET_CATALOGO = "hp9r-jxuu"

MUNICIPIOS_VALLE_ABURRA = {
    "medellin", "medellín", "bello", "itagui", "itagüi", "itagüí",
    "envigado", "sabaneta", "la estrella", "caldas", "copacabana",
    "girardota", "barbosa",
}
```

**Decisiones de diseño:**

* El punto de referencia no es el cauce exacto, sino una aproximación del barrio Sevilla / Comuna 4 Aranjuez (`6.271, -75.564`).
* Se descargan todos los municipios del Valle de Aburrá porque “el radar/pluviometría no respeta límites municipales a esta escala” — comentario propio del script.
* Normaliza nombres de columna en minúscula porque Socrata publica el esquema en minúscula; el script tolera el campo `ubicaci_n` anidado por compatibilidad con versiones anteriores de la API.

---

### 4.2 `scripts/02_pull_historico.py` — Descargar histórico de precipitación

**Qué hace paso a paso:**

1. Recibe uno o más códigos de estación por línea de comandos.
2. Inspecciona el esquema real de los datasets:
   * `s54a-sgyg` — Precipitación histórica.
   * `ksew-j3zj` — Precipitación cuasi tiempo real.
3. Detecta automáticamente el campo de código entre candidatos como `codigoestacion`, `codigo_estacion`, `estacion`, etc.
4. Prueba varias normalizaciones del código: original, sin ceros iniciales (`lstrip("0")`) y rellenado a 10 dígitos (`zfill(10)`).
5. Descarga año por año, paginando en bloques de `page_size=5000`.
6. Si un rango anual falla, lo **bisecciona recursivamente** hasta un piso de 7 días.
7. Registra rangos que no se pudieron descargar en `data/raw/GAPS_<codigo>.txt`.
8. Dedup por `[campo_codigo, fechaobservacion, codigosensor, valorobservado]`.
9. Guarda `data/raw/historico_<codigo>.csv` y, si aplica, `data/raw/tiemporeal_<codigo>.csv`.

**Entrada:** códigos de estación por CLI, por ejemplo:

```bash
python scripts/02_pull_historico.py 0027015290 0027015310 0027015330
```

**Salida:** CSVs en `data/raw/`.

**Constantes y variables clave:**

```python
DATASET_HISTORICO = "s54a-sgyg"
DATASET_TIEMPO_REAL = "ksew-j3zj"

CAMPOS_CODIGO_POSIBLES = [
    "codigoestacion", "codigo_estacion", "codigoestacacion",
    "estacion", "codigo", "codigoestac",
]

page_size = 5000
piso_dias = 7   # dentro de descarga_rango_recursiva
```

**Decisiones de diseño:**

* “No asumimos nombres de columna a ciegas, porque Socrata a veces los cambia entre datasets.”
* La descarga por años permite reanudar parcialmente sin perder todo el progreso.
* El script deja explícitas las credenciales en una variable `_partes_dominio` y usuario/contraseña; esto **no debería subirse a Git**, pero está presente en el código local.
* Si no hay datos en `s54a-sgyg` para un código, no genera un CSV vacío; advierte para evitar silencios.

**Estaciones reales descargadas en este repo:**

| Código | Nombre | Sensor canónico | Sensor QA | Rango aproximado (CSV) |
|--------|--------|-----------------|-----------|------------------------|
| `0027015290` | Pajarito | 240 | — | 2016-12-31 → 2020-03-28 |
| `0027015310` | Metromedellín | 240 | — | 2016-12-31 → 2026-07-09 |
| `0027015330` | Olaya Herrera | 240 | 257 | 2016-12-31 → 2026-07-09 |

Estos metadatos están codificados en `simulate/04_scs_cn_proxy.py`, no solo en `02_pull_historico.py`.

---

### 4.3 `simulate/cuenca_la_honda_params.py` — Parámetros de la cuenca

**Qué hace:** centraliza los parámetros morfométricos e hidrológicos y calcula el CN compuesto.

**Constantes clave:**

```python
AREA_TOTAL_KM2 = 5.9
LONGITUD_CAUCE_KM = 6.2

SUPUESTO_REPARTO_AREA = {
    "alto": 0.40,
    "medio": 0.30,
    "bajo": 0.30,
}

SUPUESTO_CN_POR_TRAMO = {
    "alto": 70,   # bosque en buena condición, HSG C
    "medio": 92,  # residencial denso informal
    "bajo": 85,   # urbano mixto
}

CN = calcular_cn_compuesto()   # 81.1
```

**Cálculo real:**

```python
cn = sum(
    SUPUESTO_REPARTO_AREA[tramo] * SUPUESTO_CN_POR_TRAMO[tramo]
    for tramo in TRAMOS
)
return round(cn, 1)   # 0.40*70 + 0.30*92 + 0.30*85 = 81.1
```

**Decisiones / advertencias del propio código:**

* `AREA_TOTAL_KM2` y `LONGITUD_CAUCE_KM` son datos oficiales AMVA.
* `SUPUESTO_REPARTO_AREA` y `SUPUESTO_CN_POR_TRAMO` son **supuestos de trabajo** sin fuente oficial.
* Asume Hydrologic Soil Group C (suelos residuales/saprolíticos). El grupo real no está confirmado con estudio de suelos IGAC/IDEAM.

---

### 4.4 `simulate/04_scs_cn_proxy.py` — Proxy SCS-CN horario

**Qué hace paso a paso:**

1. **Paso 0 — Inspección:** abre cada `historico_*.csv`, detecta columnas de sensor, precipitación y tiempo, y decide el tratamiento.
2. **Paso 1 — Limpieza por estación:**
   * Para Olaya Herrera, separa el sensor QA `257` y conserva solo el sensor principal `240`.
   * Convierte `valorobservado` a numérico y `fechaobservacion` a `datetime`.
   * Elimina duplicados.
   * Agrupa por timestamp y suma.
   * Resamplea a **1 h con `sum()`** porque los datos son incrementos de pluviómetro de balde (~10 min).
3. **Paso 2 — P_basin:** concatena las tres series horarias con `join="outer"`, rellena el índice horario completo y calcula:
   * `P_basin = mean(axis=1, skipna=True)` — promedio de estaciones disponibles.
   * `n_estaciones = df.notna().sum(axis=1)` — cuántas estaciones aportan en cada hora.
4. **Paso 3 — SCS-CN:**
   * Acumula 24 h: `p_acc_24h = p_basin.rolling(window=24, min_periods=1).sum()`.
   * Enmascara la acumulación si hay menos de 12 horas válidas en la ventana.
   * Calcula `S = (25400 / CN) - 254` e `Ia = 0.2 * S`.
   * Aplica:

```python
q.loc[no_luvia] = 0.0
q.loc[lluvia] = (exceso**2) / (exceso + s)
```

5. **Paso 4 — Guarda** `data/processed/proxy_q_la_honda.csv`.
6. **Verificación obligatoria:** imprime cobertura, % NaN, % Q=0, mínimo, máximo, media y un chequeo del hueco de Metromedellín.

**Entrada:**

```python
HISTORICO_FILES = [
    DATA_RAW_DIR / "historico_0027015290.csv",
    DATA_RAW_DIR / "historico_0027015310.csv",
    DATA_RAW_DIR / "historico_0027015330.csv",
]
```

**Salida:** `data/processed/proxy_q_la_honda.csv`.

**Constantes clave:**

```python
CN = 81.1  # importado de cuenca_la_honda_params.py

STATION_META = {
    "historico_0027015290.csv": {"codigo": "0027015290", "nombre": "Pajarito",       "sensor_principal": 240},
    "historico_0027015310.csv": {"codigo": "0027015310", "nombre": "Metromedellin",  "sensor_principal": 240},
    "historico_0027015330.csv": {"codigo": "0027015330", "nombre": "Olaya Herrera",  "sensor_principal": 240, "sensor_qa": 257},
}
```

**Resultado real del proxy (última corrida):**

* `proxy_q_la_honda.csv`: 83 458 filas horarias, desde `2016-12-31 14:00:00` hasta `2026-07-09 23:00:00`.
* Columnas: `timestamp`, `P_basin`, `P_acc_24h`, `Q_scs_proxy`, `n_estaciones_disponibles`.
* Con CN=81.1: `S ≈ 59.20`, `Ia ≈ 11.84`.

**Decisiones de diseño:**

* Se usa `sum()` en el resample porque las lecturas son **incrementos de pluviómetro** (báscula), no intensidad instantánea.
* Se promedia con `skipna=True` para no perder horas con solo 1 o 2 estaciones activas; luego el ETL decide cuáles descartar.
* El acumulado de 24 h usa `min_periods=1` pero enmascara si `validos < 12`, lo que introduce NaN en ventanas con muchos huecos.

---

### 4.5 `simulate/05_validar_proxy_eventos.py` — Validación exploratoria

**Qué hace paso a paso:**

1. Carga `data/processed/proxy_q_la_honda.csv`.
2. Si `EVENTOS_CONOCIDOS` tiene fechas externas, calcula para cada una:
   * Ventana de ±48 h.
   * Valor de `Q_scs_proxy` en el timestamp más cercano.
   * Percentil de ese valor respecto a toda la serie.
   * Umbral top 5 % (`quantile(0.95)`).
   * Bandera roja si el evento NO cae en el top 5 %.
3. Si `EVENTOS_CONOCIDOS` está vacío, advierte y omite la validación temporal.
4. Siempre grafica la serie completa con marcas verticales (aunque estén vacías) y guarda `data/processed/proxy_q_la_honda_eventos.png`.

**Entrada:** `data/processed/proxy_q_la_honda.csv`.

**Salida:** `data/processed/proxy_q_la_honda_eventos.png`.

**Constante clave:**

```python
EVENTOS_CONOCIDOS = {}
```

**Decisiones de diseño:**

* “Fechas exactas pendientes — ver PQRSD a `atencionusuario@metropol.gov.co` o fuente DAGRD directa. NO inferir de la serie propia.”
* El top 5 % se calcula sobre toda la serie; no se ajusta por estación del año.
* En el estado actual del repo no hay eventos oficiales cargados, por lo que la validación es meramente visual.

---

### 4.6 `simulate/06_etl_features.py` — ETL de features y targets

**Qué hace paso a paso:**

1. **Carga** `proxy_q_la_honda.csv`.
2. **Diagnóstico pre-filtro:**
   * Cuenta, mes a mes, cuántas horas tienen `n_estaciones_disponibles = 0, 1, 2, 3`.
   * Destaca meses donde `n=0` o `n=1` son mayoría (>50 %).
   * Cruza esos meses con dos ventanas documentadas:

```python
PAJARITO_FIN_ACTIVO         = pd.Timestamp("2020-03-28 23:59:59")
PAJARITO_SUSPENDIDO_INICIO  = pd.Timestamp("2020-03-29 00:00:00")
METROMEDELLIN_GAP_INICIO    = pd.Timestamp("2019-02-18 16:00:00")
METROMEDELLIN_GAP_FIN       = pd.Timestamp("2019-07-26 11:00:00")
```

   * Si hay meses con n<2 **fuera** de esas ventanas, los reporta como hallazgos nuevos.
3. **Filtro de calidad:** separa filas con `n_estaciones_disponibles < 2` hacia `filas_baja_calidad.csv`; el resto continúa.
4. **Base horaria:** construye un índice horario completo y rellena con NaN las horas faltantes.
5. **Features (`armar_features`):**

```python
features["P_basin"] = base_horaria["P_basin"]
features["Q_actual"] = base_horaria["Q_scs_proxy"]

for horas in (1, 3, 6, 12, 24):
    features[f"lag_{horas}h"] = base_horaria["P_basin"].shift(horas)

for horas in (1, 3, 6):
    features[f"Q_lag_{horas}h"] = base_horaria["Q_scs_proxy"].shift(horas)

for horas in (3, 6, 12, 24, 48):
    features[f"roll_sum_{horas}h"] = base_horaria["P_basin"].rolling(window=horas, min_periods=1).sum()

features["roll_max_6h"] = base_horaria["P_basin"].rolling(window=6, min_periods=1).max()
features["hora_dia"] = features.index.hour
features["mes"] = features.index.month
```

6. **Targets:** crea `target_6h`, `target_12h`, `target_24h` con `base_horaria["Q_scs_proxy"].shift(-horizonte_horas)`.
7. **Split cronológico:**
   * Si ya existe un `dataset_{h}h.csv` previo con columna `split`, **preserva el corte** para no cambiar la evaluación entre corridas.
   * Si no, usa proporción **82/18** (`int(n_total * 0.82)`).
8. **Guarda** `dataset_6h.csv`, `dataset_12h.csv`, `dataset_24h.csv`.

**Entrada:** `data/processed/proxy_q_la_honda.csv`.

**Salidas:**

* `data/processed/filas_baja_calidad.csv`
* `data/processed/dataset_6h.csv`
* `data/processed/dataset_12h.csv`
* `data/processed/dataset_24h.csv`

**Constantes y variables clave:**

```python
HORIZONTES = (6, 12, 24)
LOW_QUALITY_MIN_ESTACIONES = 2

# Split 82/18:
corte = int(n_total * 0.82)
```

**Resultado real de la última corrida:**

| Métrica | Valor |
|---------|-------|
| Filas originales | 83 458 |
| Filas baja calidad (`n<2`) | 35 980 |
| Filas tras filtro | 47 478 |
| Pérdida por `dropna` de features | 3 613 |
| Pérdida adicional por `dropna` target 6 h | 797 |
| **Dataset 6 h final** | **43 068** |
| Dataset 12 h final | 42 693 |
| Dataset 24 h final | 42 200 |
| Train 6 h | 35 315 |
| Test 6 h | 7 753 |
| `% target_6h == 0` | 91.86 % |

**Decisiones de diseño:**

* El umbral `n_estaciones_disponibles >= 2` descarta horas donde solo una estación reporta, reduciendo el riesgo de usar un único sensor con falla.
* Los lags y rolling usan desplazamientos positivos (`shift(horas)`) o ventanas hacia atrás; el target usa `shift(-horas)`. Ninguna feature mira hacia adelante, evitando fuga de datos.
* El split se preserva entre corridas para mantener comparabilidad de métricas.
* Los meses lluviosos se definen como abril, mayo, octubre y noviembre en `clasificar_meses_temporada()`; esto solo sirve para el diagnóstico, no para el modelo.

**Nota sobre las fechas de hueco:**

Las constantes `METROMEDELLIN_GAP_INICIO`/`METROMEDELLIN_GAP_FIN` y `PAJARITO_SUSPENDIDO_INICIO` no se usan para **mascarar** datos; solo se usan en `diagnosticar_disponibilidad_pre_filtro()` para explicar meses con baja disponibilidad. El filtro real es `n_estaciones_disponibles < 2`.

---

## 5. Modelo predictivo

### 5.1 RandomForestRegressor — los 3 intentos descartados

`simulate/07_entrenar_modelo_6h.py` entrena y compara tres regresores contra un **baseline de persistencia** (`Q_scs_proxy` actual como predicción de 6 h futuro).

**Features usadas (18):**

```python
['P_basin', 'Q_actual',
 'lag_1h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h',
 'Q_lag_1h', 'Q_lag_3h', 'Q_lag_6h',
 'roll_sum_3h', 'roll_sum_6h', 'roll_sum_12h', 'roll_sum_24h', 'roll_sum_48h',
 'roll_max_6h', 'hora_dia', 'mes']
```

**Hiperparámetros comunes:**

```python
RandomForestRegressor(
    random_state=42,
    n_estimators=200,
    n_jobs=-1,
)
```

**Enfoques:**

| Enfoque | Target de entrenamiento | Peso |
|---------|------------------------|------|
| Target absoluto | `target_6h` | uniforme |
| Weighted | `target_6h` | `sample_weight = 1.0 + (y_train / y_train_max) * 20.0` |
| Delta residual | `delta_6h = target_6h - Q_actual` | uniforme; luego se reconstruye `pred_absoluta = Q_actual + pred_delta` |

**Resultados reales de la última corrida (2026-07-27):**

#### Global

| Enfoque | MAE | RMSE | R² |
|---------|-----|------|----|
| Baseline persistencia | 0.014288 | 0.139495 | 0.510524 |
| RF target absoluto | 0.036479 | 0.175972 | 0.221064 |
| RF weighted (×20) | 0.041974 | 0.207561 | -0.083693 |
| RF delta reconstruido | 0.038530 | 0.183886 | 0.149419 |

#### Subset eventos (`target_6h > 0`, n=366)

| Enfoque | MAE | RMSE | R² |
|---------|-----|------|----|
| Baseline persistencia | 0.189361 | 0.496568 | 0.600092 |
| RF target absoluto | 0.350105 | 0.613250 | 0.390074 |
| RF weighted (×20) | 0.374840 | 0.643075 | 0.329303 |
| RF delta reconstruido | 0.370445 | 0.634565 | 0.346936 |

**Por qué fallaron (según el propio código y salida):**

* El dataset tiene ~91.86 % de filas con `target_6h == 0` y solo 366 eventos (>0) en test (~4.7 %).
* RandomForest promedia hacia la media y diluye la cola de eventos.
* El script imprime explícitamente:

```text
ADVERTENCIA EXPLICITA: RF NO supera al baseline de persistencia en el subset de eventos (target_6h > 0).
```

* El baseline de persistencia es un predictor fuerte para series con alta autocorrelación horaria: si ahora hay escorrentía, en 6 h probablemente siga parecida.

**Archivos generados (NO trackeados):**

Los tres regresores se guardaban como `rf_6h.joblib` (~61 725 KB), `rf_6h_delta.joblib` (~74 161 KB) y `rf_6h_weighted.joblib` (~71 047 KB). **Ya no existen en disco**: fueron eliminados por limpieza en el commit `4b747e3`; solo se conserva `clf_6h.joblib` (el modelo en producción). Si se vuelven a necesitar, `simulate/07_entrenar_modelo_6h.py` los regenera.

---

### 5.2 RandomForestClassifier — modelo en producción

`simulate/08_clasificador_6h.py` entrena un clasificador binario (`target_6h > p90`). El dashboard (`dashboard/app.py`) interpreta la probabilidad bruta con una escala ternaria:

| Estado | Rango de probabilidad | Color |
|--------|----------------------|-------|
| NORMAL | `proba < 0.30` | verde (#2ecc71) |
| PRECAUCIÓN | `0.30 ≤ proba < 0.70` | amarillo (#f39c12) |
| ALERTA | `proba ≥ 0.70` | rojo (#e74c3c) |

**Target binario (entrenamiento):**

```python
p90 = float(train[TARGET_COL].quantile(0.90))
train["label_6h"] = (train[TARGET_COL] > p90).astype(int)
test["label_6h"]  = (test[TARGET_COL] > p90).astype(int)
```

Con la distribución actual, `p90 = 0.0`, por lo que la clase positiva equivale a `target_6h > 0`.

**Hiperparámetros exactos:**

```python
RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    n_jobs=-1,
    class_weight="balanced",
)
```

**Entrenamiento:**

* `n train = 35 315`, `n test = 7 753`.
* Positivos train: 3 138 (8.89 %).
* Positivos test: 366 (4.72 %).

**Resultados en test con threshold por defecto (0.5):**

```text
              precision    recall  f1-score   support

           0   0.980514  0.980912  0.980713      7387
           1   0.611570  0.606557  0.609053       366

    accuracy                       0.963240      7753
```

**Barrido de umbrales sobre `predict_proba` (clase positiva) — referencia del modelo binario:**

| Umbral | Recall | Precisión | FN | FP | TP |
|--------|--------|-----------|----|----|----|
| 0.50 | 0.606557 | 0.611570 | 144 | 141 | 222 |
| 0.40 | 0.658470 | 0.539150 | 125 | 206 | 241 |
| 0.30 | 0.702186 | 0.476809 | 109 | 282 | 257 |
| 0.20 | 0.743169 | 0.400000 | 94 | 408 | 272 |
| 0.15 | 0.781421 | 0.311208 | 80 | 633 | 286 |
| 0.10 | 0.816940 | 0.256873 | 67 | 865 | 299 |

**Observación importante sobre el recomendador automático:**

El script `08_clasificador_6h.py` tiene una función `barrer_umbrales()` que recomienda automáticamente el umbral más bajo con `recall >= 0.85`. En la última corrida **ningún umbral en [0.1, 0.5] alcanza ese recall**, por lo que el script imprime:

```text
LIMITACION A DOCUMENTAR: ningun umbral del barrido alcanza recall >= 0.85.
El maximo recall alcanzable en [0.1, 0.5] es 0.816940 con umbral=0.10
```

El dashboard expone tres niveles de alerta (NORMAL / PRECAUCIÓN / ALERTA) sobre la misma probabilidad binaria. El umbral de ALERTA (0.70) es más conservador que el umbral binario histórico (0.20), mientras que PRECAUCIÓN (0.30) actúa como zona de vigilancia. El clasificador sigue siendo **una fuente dentro de una fusión multi-sensor**; se acepta más falsos positivos a cambio de no perder eventos.

**Feature importance del clasificador (ordenada):**

```text
roll_sum_24h: 0.206408
roll_sum_48h: 0.127721
Q_actual:     0.118186
Q_lag_1h:     0.109794
roll_sum_12h: 0.080958
hora_dia:     0.045520
Q_lag_3h:     0.045240
roll_max_6h:  0.044988
roll_sum_6h:  0.037342
mes:          0.035238
...
```

**Archivo generado (NO trackeado):**

```text
simulate/models/clf_6h.joblib   # 42 463 KB
```

### 5.3 SMOTENC

No se usó SMOTENC ni ninguna técnica SMOTE. El balanceo se hace únicamente con `class_weight="balanced"`.

---

## 6. Simulador de gateway

`simulate/09_gateway_simulado.py` replica el comportamiento que tendría un gateway RPi4 leyendo la serie fila a fila.

### Qué valida exactamente

1. **Motor de features en streaming:** calcula las mismas 18 features que `06_etl_features.py`, pero usando `collections.deque(maxlen=72)` en lugar de vectores de pandas.
2. **Sin fuga de datos:** las funciones `_lag()` y `_rolling_sum()` solo miran hacia atrás en el tiempo.
3. **Alineación batch vs streaming:** compara 500 timestamps aleatorios del `dataset_6h.csv` contra los valores calculados en streaming; tolerancia:

```python
EPSILON = 1e-6
N_VALIDACION_TS = 500
```

4. **Inferencia por micro-lotes:** procesa features en lotes de 2048 y predice con `n_jobs=1` para evitar overhead de threading.
5. **Umbral de alerta (gateway):** `UMBRAL_ALERTA = 0.20` (binario en `09_gateway_simulado.py`). El dashboard reinterpreta esa probabilidad en tres estados: NORMAL `< 0.30`, PRECAUCIÓN `[0.30, 0.70)`, ALERTA `≥ 0.70`.

### Metodología

```python
def calcular_features_streaming(base_stream):
    p_hist = deque(maxlen=72)
    q_hist = deque(maxlen=72)
    registros = []
    for row in base_stream.itertuples(index=False):
        p_hist.append(p_now)
        q_hist.append(q_now)
        feat = {
            "timestamp": ts,
            "P_basin": p_now,
            "Q_actual": q_now,
            "lag_1h": _lag(p_hist, 1),
            ...
            "hora_dia": int(ts.hour),
            "mes": int(ts.month),
        }
```

### Resultado de la última corrida

```text
Proxy cargado: ... | filas=83458
Dataset batch cargado: ... | filas=43068
VALIDACION OK: 500 timestamps aleatorios comparados, sin diferencias > 1e-06.
Filas inferidas (sin NaN en features): 43865
Total ALERTA_6H: 4947
Total NORMAL: 38918
```

**Distribución anual de alertas:**

| Año | ALERTA_6H | NORMAL |
|-----|-----------|--------|
| 2017 | 943 | 6 497 |
| 2018 | 1 359 | 7 164 |
| 2019 | 709 | 3 744 |
| 2020 | 112 | 1 733 |
| 2021 | 324 | 4 550 |
| 2022 | 256 | 3 568 |
| 2023 | 83 | 472 |
| 2024 | 139 | 963 |
| 2025 | 609 | 6 510 |
| 2026 | 413 | 3 717 |

**Salida:** `data/processed/log_gateway_simulado.csv`.

Columnas del log:

```text
timestamp,P_basin,Q_actual,Q_lag_1h,roll_sum_24h,hora_dia,mes,proba_alerta,estado
```

---

## 7. Dashboard

### 7.1 `dashboard/app.py` — Aplicación Streamlit

**Lanzamiento:**

```bash
streamlit run dashboard/app.py
```

**Datos que consume:**

* `data/processed/log_gateway_simulado.csv`
* `data/processed/proxy_q_la_honda.csv`
* `simulate/models/clf_6h.joblib`

**Constantes clave:**

```python
RECALL_OPERATIVO = 0.743
PRECISION_OPERATIVA = 0.400
UMBRAL_PRECAUCION = 0.30
UMBRAL_ALERTA = 0.70

N_NODOS = 5

FEATURE_COLS = [
    "P_basin", "Q_actual",
    "lag_1h", "lag_3h", "lag_6h", "lag_12h", "lag_24h",
    "Q_lag_1h", "Q_lag_3h", "Q_lag_6h",
    "roll_sum_3h", "roll_sum_6h", "roll_sum_12h", "roll_sum_24h", "roll_sum_48h",
    "roll_max_6h", "hora_dia", "mes",
]

ANCLAS = [
    {"nombre": "Cuenca Alta (Parque Arví)",      "lat": 6.2804, "lon": -75.5027},
    {"nombre": "Tramo Medio (La Honda)",  "lat": 6.2694, "lon": -75.5648},
    {"nombre": "Tramo Bajo (Moravia / Montecarlo)", "lat": 6.2781, "lon": -75.5670},
]
```

### Vista Pública

* **Semáforo + probabilidad:** muestra el último estado del log (`NORMAL`, `PRECAUCIÓN` o `ALERTA`) y `proba_alerta`. Estados: NORMAL `proba < 0.30` (verde), PRECAUCIÓN `[0.30, 0.70)` (amarillo), ALERTA `≥ 0.70` (rojo).
* **Tarjetas de lluvia acumulada:**
  * Últimas 24 h.
  * Últimos 30 días.
  * Año en curso.
  * La referencia es siempre `proxy["timestamp"].max()` (actualmente `2026-07-09`), no la fecha del reloj del sistema.
* **Mapa de nodos:**
  * Interactivo con Folium si hay internet.
  * Fallback estático con Matplotlib si no hay conexión.
  * 5 nodos interpolados por longitud de arco entre las 3 anclas.
* **Calendario de alertas:** cuadrícula de los últimos 90 días; verde = solo NORMAL, amarillo = al menos una hora PRECAUCIÓN, rojo = al menos una hora ALERTA.

### Vista Operador (JAC)

* **Log de gateway:** tabla con las últimas 100 filas de `log_gateway_simulado.csv`.
* **Gráfico histórico dual-axis:**
  * `P_basin` (eje izquierdo).
  * `Q_scs_proxy` + alertas (eje derecho).
  * Selector de rango de fechas.
* **Métricas del modelo:** recall=0.743, precisión=0.400 (referencia del modelo binario); umbrales ternarios PRECAUCIÓN=0.30, ALERTA=0.70.
* **Feature importance:** barras horizontales de `clf_6h.joblib`.
* **Inspección de punto histórico:** selección de fecha y hora para ver `estado`, `proba_alerta` y features.
* **Simulador interactivo del modelo.**

### 7.2 Simulador interactivo

Se encuentra en `_simulador_modelo()` dentro de `dashboard/app.py`.

**3 presets exactos:**

```python
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
```

**Features derivadas aproximadas (no exactas, documentadas en el propio dashboard):**

```python
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
```

> El preset “Lluvia fuerte sostenida” arroja una probabilidad real de aproximadamente **0.965** (96.5 %) con `hora_dia=14, mes=7`, según `dashboard/_check_fixes.py`. Eso dispara el estado **ALERTA** (umbral ternario `≥ 0.70`) con una confianza mucho mayor que ~77 %.

### 7.3 `dashboard/_check_fixes.py`

Script de verificación que:

1. Confirma que `_lluvia_acumulada()` usa `proxy["timestamp"].max()` y no la fecha del sistema.
2. Confirma que `_mapa_folium()` usa `fit_bounds` y `pad = 0.015`.
3. Confirma que existen `_simulador_modelo()` y `_ESCENARIOS` con 3 escenarios.
4. Calcula la probabilidad del preset “Lluvia fuerte” y del escenario normal.
5. Verifica que `vista_operador()` llama al simulador.

---

## 8. Cómo correr el proyecto desde cero

### 8.1 Entorno

Se recomienda usar el `venv` ya existente en el repo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& C:\Users\Nico\Desktop\CentineLA\venv\Scripts\Activate.ps1
```

O crear uno nuevo e instalar dependencias. El único archivo de requerimientos es `requirements.txt` en la raíz; incluye las versiones exactas del entorno de entrenamiento y del dashboard.

```bash
pip install -r requirements.txt
```

### 8.2 Orden de ejecución

**Fase 1 — Extracción IDEAM (requiere internet):**

```bash
python scripts/01_buscar_estaciones.py
python scripts/02_pull_historico.py 0027015290 0027015310 0027015330
```

**Fase 2 — Pipeline de datos y modelos:**

```bash
python simulate/04_scs_cn_proxy.py
python simulate/05_validar_proxy_eventos.py
python simulate/06_etl_features.py
python simulate/07_entrenar_modelo_6h.py
python simulate/08_clasificador_6h.py
python simulate/09_gateway_simulado.py
```

**Fase 3 — Dashboard:**

```bash
streamlit run dashboard/app.py
```

### 8.3 Variables de entorno

El proyecto usa un archivo `.env` en la raíz del repositorio (no trackeado por Git). Hay un archivo de ejemplo trackeado:

```bash
cp .env.example .env
```

En Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edita `.env` y completa al menos estas variables antes de ejecutar `02_pull_historico.py`:

```text
SODAPY_USERNAME=<tu-usuario-de-socrata>
SODAPY_PASSWORD=<tu-contraseña-de-socrata>
```

**Obligatorias para Fase 1:**

| Variable | Script que la usa | Descripción |
|----------|-------------------|-------------|
| `SODAPY_USERNAME` | `scripts/02_pull_historico.py` | Usuario de la cuenta Socrata en datos.gov.co |
| `SODAPY_PASSWORD` | `scripts/02_pull_historico.py` | Contraseña de la cuenta Socrata |

**Opcionales:**

| Variable | Script que la usa | Descripción |
|----------|-------------------|-------------|
| `SODAPY_APP_TOKEN` | `scripts/02_pull_historico.py` | Token opcional de Socrata para evitar rate limits estrictos |
| `WIFI_SSID` | Nodos IoT (futuro) | Red WiFi del prototipo ESP32 |
| `WIFI_PASSWORD` | Nodos IoT (futuro) | Contraseña de red WiFi del prototipo |
| `UBIDOTS_TOKEN` | Nodos IoT (futuro) | Token de Ubidots para publicación MQTT |

Si alguna variable obligatoria falta, `02_pull_historico.py` lanza:

```text
ValueError: SODAPY_USERNAME y SODAPY_PASSWORD deben estar definidos en el entorno. Copia .env.example a .env y completa los valores.
```

### 8.4 Seguridad y rotación de credenciales

- **Nunca subas `.env` a Git.** Ya está incluido en `.gitignore`.
- Si sospechas que las credenciales Socrata se expusieron (por ejemplo, en un commit anterior), rótalas desde el panel de Socrata / datos.gov.co y actualiza solo tu `.env` local.
- Si trabajas en equipo, comparte los valores por un canal seguro (gestor de contraseñas, variables de CI/CD), nunca por el repositorio.

---

## 9. Limitaciones conocidas

Solo se incluyen limitaciones confirmadas por el código o los datos:

1. **Proxy regional, no local.** Las estaciones más cercanas con histórico útil (Pajarito, Metromedellín, Olaya Herrera) están a 5.7–6.7 km de La Honda. No hay estación in-situ en la quebrada.
2. **Base de eventos históricos reducida.** `EVENTOS_CONOCIDOS = {}`; no hay fechas oficiales de avenidas torrenciales cargadas para validar el proxy.
3. **CN estimado.** El CN compuesto de 81.1 depende de un reparto de área por tramo (40/30/30) y CN por tramo (70/92/85) que son supuestos de trabajo sin fuente oficial.
4. **SCS-CN aplicado como evento único sobre serie continua.** El proxy usa acumulado 24 h sin reiniciar explícitamente eventos de lluvia; es una aproximación documentada.
5. **Desbalance extremo.** Con `p90=0.0`, la clase positiva equivale a `target_6h > 0`, que representa ~8.9 % en entrenamiento y ~4.7 % en test.
6. **No hay split aleatorio; es cronológico.** El test está fijado desde `2025-07-06 16:00` en adelante. Cualquier cambio estructural en los datos recientes afecta las métricas.
7. **Recall aislado del clasificador no supera ~0.82.** A umbral 0.10 se alcanza 0.817; a 0.20 es 0.743. El diseño asume que el clasificador será una de varias fuentes en una regla de fusión multi-sensor. El estado PRECAUCIÓN (0.30–0.70) amplía la vigilancia sin disparar la alerta máxima.
8. **Modelos y datos grandes están parcialmente versionados.** Desde la limpieza de 2026-10, los datasets procesados y `clf_6h.joblib`/`clf_6h_baseline_20260913.joblib` están en el repo; los `rf_6h*.joblib` y `data/raw/*.csv` se regeneran o comparten por otro medio.
9. **Capa física (LoRa, sensores, cámaras) no implementada.** Solo existe el simulador y la arquitectura documentada.
10. **Prototipo IoT base documentado solo como referencia.** Los issues del sketch Wokwi (credenciales WiFi expuestas, `WiFiMulti`, umbral de temperatura) no se han resuelto en este repo.

---

## 10. Estado del repositorio (git)

### Sí es un repositorio Git

A diferencia de lo que afirma `STATUS_REPORT.md` (auditoría 2026-07-21), el directorio **sí es un repositorio Git** en la fecha actual. El historial fue **reescrito** (`e8d53c4` “CentineLA: historial limpio”) para eliminar credenciales Socrata que estuvieron expuestas en commits antiguos, por lo que los commits anteriores a la reescritura **ya no existen**. Historial actual completo:

```text
$ git log --oneline
4b747e3 chore: limpia modelos huérfanos rf_6h y directorio basura {data/
a28613e fix: corrige etiqueta "Tramo Medio" (Jardin Botanico -> La Honda) en dashboard y README, coincide con TRAMOS real
5a4324b docs: update outdated markdown files to reflect current repo state
629e887 docs: document .env setup and credential rotation
e8d53c4 CentineLA: historial limpio
```

### Archivos trackeados

```text
.env.example
.gitignore
.vscode/settings.json
CENTINELA_CONTEXTO_TECNICO.md
README.md
RESUMEN_SESION_CENTINELA_2026-07-11.md
STATUS_REPORT.md
dashboard/_check_fixes.py
dashboard/app.py
requirements.txt
data/processed/dataset_12h.csv
data/processed/dataset_24h.csv
data/processed/dataset_6h.csv
data/processed/filas_baja_calidad.csv
data/processed/log_gateway_simulado.csv
data/processed/proxy_q_la_honda.csv
data/processed/proxy_q_la_honda_eventos.png
scripts/01_buscar_estaciones.py
scripts/02_pull_historico.py
simulate/04_scs_cn_proxy.py
simulate/05_validar_proxy_eventos.py
simulate/06_etl_features.py
simulate/07_entrenar_modelo_6h.py
simulate/08_clasificador_6h.py
simulate/09_gateway_simulado.py
simulate/cuenca_la_honda_params.py
simulate/models/clf_6h.joblib
simulate/models/clf_6h_baseline_20260913.joblib
simulate/models/rf_6h.joblib
simulate/models/rf_6h_delta.joblib
simulate/models/rf_6h_weighted.joblib
```

### Archivos NO trackeados (`.gitignore`)

```gitignore
__pycache__/
*.pyc
venv/
venv_test/
.venv/
data/raw/*.csv
```

Por tanto, los siguientes artefactos **no están en Git** y deben regenerarse o transferirse por otro canal:

* `data/raw/historico_*.csv`
* `data/raw/estaciones_candidatas.csv`
* `data/processed/*.csv`
* `simulate/models/*.joblib`

### Cómo regenerar lo que no está trackeado

1. Ejecutar `scripts/01_buscar_estaciones.py` y `scripts/02_pull_historico.py` para regenerar `data/raw/`.
2. Ejecutar los scripts `04` al `09` de `simulate/` para regenerar `data/processed/` y `simulate/models/`.
3. Ejecutar `streamlit run dashboard/app.py` para levantar el dashboard.

---

## 11. Discrepancias encontradas

Se listan diferencias entre el código real y documentación o comentarios previos:

1. **`STATUS_REPORT.md` dice que no es repositorio Git, pero sí lo es.**
   * `STATUS_REPORT.md` línea 5: “el directorio no es un repositorio Git (`git status` devuelve `fatal: not a git repository`)”.
   * Realidad actual: `git log` muestra el historial reescrito a partir de `e8d53c4` (“CentineLA: historial limpio”) y `git ls-files` devuelve los archivos trackeados actuales.

2. **`STATUS_REPORT.md` reporta fechas erróneas de hueco de Metromedellín en `06_etl_features.py`, pero el código actual está corregido.**
   * `STATUS_REPORT.md` línea 19: afirma que `METROMEDELLIN_GAP_INICIO = 2019-07-01 00:00:00` y `METROMEDELLIN_GAP_FIN = 2020-12-31 23:59:59`.
   * Código actual:

```python
METROMEDELLIN_GAP_INICIO = pd.Timestamp("2019-02-18 16:00:00")
METROMEDELLIN_GAP_FIN    = pd.Timestamp("2019-07-26 11:00:00")
```

   * La corrección quedó absorbida en la reescritura de historial (`e8d53c4`); los commits que la introdujeron (`dddfc2b` y anteriores) ya no existen en el historial actual.

3. **`STATUS_REPORT.md` dice que `requirements.txt` raíz es incompleto, pero el archivo actual está completo.**
   * `STATUS_REPORT.md` líneas 147-157: reporta que faltaban `numpy`, `matplotlib`, `plotly`, `folium`, `streamlit-folium` y `requests`.
   * `requirements.txt` actual incluye todas esas dependencias.
   * El `requirements.txt` raíz ya está completo en el historial actual; la corrección de dependencias también quedó absorbida en la reescritura `e8d53c4`.

4. **`CENTINELA_CONTEXTO_TECNICO.md` ubica la estructura del repo bajo `centinela-demo/`, pero el repo real usa `CentineLA/` como raíz.**
   * La sección “Estructura del repo (`centinela-demo/`)” muestra carpetas `centinela-demo/scripts/`, `centinela-demo/data/`, etc.
   * Realmente los scripts, datos y modelos están directamente bajo `CentineLA/`, no anidados en `centinela-demo/`.
   * Además, la carpeta `centinela-demo/` y su contenido fueron eliminados del repositorio en esta sesión.

5. **`CENTINELA_CONTEXTO_TECNICO.md` dice que `07_entrenar_modelo_6h.py` contiene el clasificador RF, pero el clasificador está en `08_clasificador_6h.py`.**
   * Línea 25: “`simulate/07_entrenar_modelo_6h.py` — 3 enfoques RF regresor + RF clasificador; joblib dump de los 3 modelos.”
   * El script `07` solo entrena regresores y genera `rf_6h*.joblib`. El clasificador se entrena y guarda en `08_clasificador_6h.py` como `clf_6h.joblib`.

6. **`08_clasificador_6h.py` barre umbrales binarios (0.10–0.50); el dashboard usa dos umbrales ternarios.**
   * El recomendador interno prioriza `recall >= 0.85`; como ningún umbral del barrido lo alcanza, no recomienda un valor único.
   * `09_gateway_simulado.py` sigue usando `UMBRAL_ALERTA = 0.20` (binario); `dashboard/app.py` expone tres estados con `UMBRAL_PRECAUCION = 0.30` y `UMBRAL_ALERTA = 0.70` sobre la misma probabilidad. Las métricas de validación (recall/precisión) corresponden al modelo binario y no se han recalculado aún para los estados ternarios.

7. **Preset “Lluvia fuerte sostenida” del dashboard no produce ~77 %.**
   * `STATUS_REPORT.md` línea 84-88 documenta que alguien esperaba ~77 %.
   * `dashboard/_check_fixes.py` y la ejecución real arrojan ~0.965 (96.5 %) con `hora_dia=14, mes=7`.
   * No hay documentación previa en el repo que fije 77 % como objetivo; parece una expectativa externa incorrecta.

---

## Apéndice: números de referencia rápida

| Concepto | Valor exacto | Fuente en el código |
|----------|--------------|---------------------|
| Área cuenca | 5.9 km² | `simulate/cuenca_la_honda_params.py` |
| Longitud cauce | 6.2 km | `simulate/cuenca_la_honda_params.py` |
| CN compuesto | 81.1 | `simulate/cuenca_la_honda_params.py` |
| Reparto área alto/medio/bajo | 0.40 / 0.30 / 0.30 | `simulate/cuenca_la_honda_params.py` |
| CN por tramo alto/medio/bajo | 70 / 92 / 85 | `simulate/cuenca_la_honda_params.py` |
| Mínimo estaciones | 2 | `LOW_QUALITY_MIN_ESTACIONES = 2` en `06` y `09` |
| Horizontes generados | 6, 12, 24 h | `HORIZONTES = (6, 12, 24)` en `06` |
| Split train/test | 82 / 18 cronológico | `06_etl_features.py` |
| Estaciones usadas | 0027015290, 0027015310, 0027015330 | `04_scs_cn_proxy.py` |
| Sensor canónico | 240 | `04_scs_cn_proxy.py` |
| Sensor QA Olaya | 257 | `04_scs_cn_proxy.py` |
| Sistema de alertas | NORMAL / PRECAUCIÓN / ALERTA | `dashboard/app.py` |
| Umbral PRECAUCIÓN | 0.30 | `dashboard/app.py` (`UMBRAL_PRECAUCION`) |
| Umbral ALERTA | 0.70 | `dashboard/app.py` (`UMBRAL_ALERTA`) |
| Umbral binario gateway | 0.20 | `09_gateway_simulado.py` |
| Recall binario (umbral 0.20) | 0.752998 | `08_clasificador_6h.py` (barrido) |
| Precisión binaria (umbral 0.20) | 0.415344 | `08_clasificador_6h.py` (barrido) |
| Gateway validación | 500 timestamps, ε=1e-6 | `09_gateway_simulado.py` |
| Alertas/NORMAL gateway | 5 028 / 40 167 | `09_gateway_simulado.py` |
