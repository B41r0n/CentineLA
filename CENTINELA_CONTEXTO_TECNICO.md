# CentineLA — Contexto técnico para asistente de código

Sistema de alerta temprana de crecientes súbitas, Quebrada La Honda, Comuna 4 Aranjuez, Medellín. Proyecto individual de Bairon Calle Rivera (ITM, Territorio INN 2026). Este doc es contexto de grounding para Copilot/asistente — no es el documento de entrega del proyecto.

## Estado actual — Fase 1 CERRADA

Las 3 estaciones IDEAM objetivo tienen histórico completo, verificado fila por fila (no solo por conteo total), descargado vía scripts/02_pull_historico.py con paginación por año + bisección recursiva + dedup:

- **Pajarito (0027015290)** — 159.549 filas, 2016-12-31 a 2020-03-28.
  Estación dejó de reportar en esa fecha (catálogo dice "Suspendida", consistente). Sensor único 240. Sirve para entrenar con histórico, no para el gateway en tiempo real.
- **Metromedellín (0027015310)** — 229.535 filas, 2016-12-31 a 2026-07-09.
  Hueco real de instrumento confirmado contra CSV crudo en Fase 2: última lectura antes del hueco **2019-02-18 16:00**, primera lectura posterior **2019-07-26 11:00**. La estimación de Fase 1 ("jul-2019→dic-2020") era incorrecta — el inicio real del hueco es febrero de 2019. Marcar esa ventana como "sin dato" en el ETL, no interpolar.
- **Aeropuerto Olaya Herrera (0027015330)** — 918.046 filas, 2016-12-31 a 2026-07-09. Dos sensores de precipitación mezclados: codigosensor 240 y 257. **Decisión Fase 2: sensor 240 canónico del pipeline; 257 queda solo como QA.** Olaya sensor 240 también tiene hueco real confirmado contra CSV crudo: cae a 761 filas en 2019-03 (vs ~4000 en meses normales), en 0 durante abr-jun 2019, retoma 2019-08-24. Este hueco se detectó durante el filtro de calidad del ETL (36k filas perdidas por n_estaciones<2), no en la revisión original de Fase 1.

Las 3 están a 5,7-6,7 km de La Honda (no <1km como los candidatos originales Jardín Botánico/Bermejala, descartados por no tener histórico en s54a-sgyg por numeración de estación incompatible).

Parámetros de cuenca (Fase 1, punto 3) también cerrados — ver nueva sección "Parámetros de cuenca La Honda" más abajo.

Fase 2 CERRADA. Scripts implementados y ejecutados:

- `simulate/04_scs_cn_proxy.py` — genera proxy SCS-CN horario (P_basin→Q_scs_proxy); guarda `data/processed/proxy_q_la_honda.csv`.
- `simulate/05_validar_proxy_eventos.py` — validación exploratoria del proxy contra eventos conocidos; modo exploratorio si no hay fechas externas (evita inferencia circular).
- `simulate/06_etl_features.py` — ETL de features/targets con lags, rolling sums, Q_actual/Q_lags, diagnóstico pre-filtro mensual y split cronológico preservado entre corridas.
- `simulate/07_entrenar_modelo_6h.py` — 3 enfoques de RF regresor (target absoluto, delta, weighted) + comparación contra baseline de persistencia; joblib dump de los 3 modelos. El clasificador no está en este script; está en `simulate/08_clasificador_6h.py`.
- `simulate/08_clasificador_6h.py` — clasificador binario RF con barrido de umbrales 0.5→0.1 sobre probabilidades. El dashboard aplica sobre esa probabilidad una escala ternaria: NORMAL `<0.30`, PRECAUCIÓN `[0.30, 0.70)`, ALERTA `≥0.70`.
- `simulate/09_gateway_simulado.py` — simula gateway RPi4 fila-a-fila, valida features streaming vs batch (epsilon 1e-6), genera `log_gateway_simulado.csv`.
- `data/processed/filas_baja_calidad.csv` — filas con n_estaciones<2, conservadas para auditoría QA.

**Estaciones IDEAM elegidas para Fase 1 cerrada:**
- `0027015290` — PAJARITO - AUT — Climatológica Principal
- `0027015310` — METROMEDELLIN - AUT
- `0027015330` — AEROPUERTO OLAYA HERRERA / APTO OLAYA HERRERA - TX GPRS

**Nota de cambio de estación:** Jardín Botánico Vaisala Aut y La Bermejala Pluviómetro Aut quedaron como candidatos iniciales de cercanía, pero se sacaron de la lista final porque no aportaban histórico útil en `s54a-sgyg` para la Fase 1.

## Parámetros de cuenca La Honda

Fuente: AMVA, ficha técnica oficial (metropol.gov.co/noticias/quebrada-la-honda) y PMI Microcuenca La Honda (Plan Quebradas).

- Área: 5,9 km² (dato oficial AMVA)
- Longitud cauce principal: 6,2 km (dato oficial AMVA, confirmado en 2 fuentes AMVA)
- Límites: norte=El Molino, sur=Santa Elena, oriente=Piedras Blancas, occidente=río Aburrá
- Tramo alto: nacimiento→límite rural/urbano, Parque Arví, pendiente >30%, bosque buena cobertura
- Tramo medio: barrio La Honda→cra 45, pendiente 12-25%, urbanización densa sin planificación, **amenaza alta por avenidas torrenciales según AMVA**
- Tramo bajo: cra 45→desembocadura río Aburrá, pendiente 6-12%, urbano + corredor Jardín Botánico
- CN compuesto de trabajo: 81,1 — el reparto de área por tramo (40/30/30) y el CN por tramo (70/92/85) son SUPUESTO sin fuente oficial, no confundir con área/longitud que sí son dato duro AMVA. Ver simulate/cuenca_la_honda_params.py, marcado dato-duro-vs-supuesto explícitamente.

## Arquitectura del sistema (visión completa, para cuando se implemente hardware)

```
CUENCA ALTA                    TRAMO MEDIO                  TRAMO BAJO
(nacimiento La Honda)          (Las Granjas)                (Parque Montecarlo)

[Pluviómetro tipping bucket]  [Cámara IP + CV]             [Cámara IP + CV]
+ LoRa TX                     + Sensor turbidez óptico      + Sensor turbidez óptico
                               + Sensor ultrasónico JSN-SR04T + Sensor ultrasónico
                               + LoRa TX                    + LoRa TX
         │                           │                            │
         └───────────────────────────┴────────────────────────────┘
                                     │ LoRa 915 MHz (ISM Colombia)
                          ┌──────────▼──────────┐
                          │   GATEWAY            │
                          │   Raspberry Pi 4 4GB │
                          │   + HAT LoRa RAK2287 │
                          │   + OpenCV + Python  │
                          │   + Modelo ML sklearn│
                          │   + Broker MQTT      │
                          └──────────┬───────────┘
          ┌──────────────────────────┼─────────────────────────┐
          ▼ LoRa broadcast           ▼ Internet (4G backup)    │
   [Postes: LED semáforo    [WhatsApp / DAGRD si rojo]  [Feed JSON público → SIATA]
    360° + sirena 110dB]    [Dashboard web]
```

**Fusión de datos → decisión:** el modelo pondera cada fuente. Regla de alerta roja: ≥2 fuentes críticas correlacionadas (elimina falsos positivos). 3 estados: verde/amarillo/rojo.

## Componente IA/ML (decisión de diseño — actualizada Fase 2)

- **Pivote de regresión a clasificación:** se probaron 3 enfoques de `RandomForestRegressor` — ninguno superó el baseline de persistencia simple en el subset de eventos reales. Causa: 366 eventos en 43k filas (0.85%), el RF promedia hacia la media y diluye la cola. Decisión: pivote a `RandomForestClassifier`.
- **Target de clasificación:** `label_6h = 1` si `target_6h > p90(train)`. p90 calculado **solo sobre train**, nunca sobre test. Con distribución actual p90=0.0, equivale a "hora con escorrentía >0".
- **Algoritmo en producción:** `RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')`. `class_weight='balanced'` es la corrección estándar de sklearn para desbalance.
- **Umbral operativo del gateway (binario):** 0.20 sobre `predict_proba` en `09_gateway_simulado.py`. Recall=0.74, FP/TP≈1.5:1.
- **Sistema de alertas del dashboard (ternario, desde 2026-10):** NORMAL `proba < 0.30` (verde #2ecc71), PRECAUCIÓN `0.30 ≤ proba < 0.70` (amarillo #f39c12), ALERTA `proba ≥ 0.70` (rojo #e74c3c) — ver `clasificar_estado()` y `UMBRAL_PRECAUCION`/`UMBRAL_ALERTA` en `dashboard/app.py`. El log histórico `ALERTA_6H`/`NORMAL` se normaliza a ternario al cargar. El clasificador sigue siendo una fuente dentro de la fusión multi-sensor (regla: ≥2 fuentes críticas = rojo), no árbitro único.
- **Techo real de recall:** ~0.82 con umbral 0.10. Ningún umbral en [0.1, 0.5] alcanza recall ≥ 0.85 — limitación conocida, mitigada por diseño multi-fuente.
- **Split:** cronológico con corte fijo. Train hasta 2025-07-06 15:00 (35,315 filas, 79.6%), test desde 2025-07-06 16:00 hasta 2026-09-30 (9,051 filas, 20.4%). Nunca aleatorio.
- **Horizonte priorizado:** 6h (cuenca de ladera, tiempo de concentración corto; deadline oct-2026). Datasets 12h/24h generados pero no entrenados con el mismo rigor.
- **Métricas clave:** recall y precision de clase positiva. Accuracy es inútil con 95% de clase negativa — baseline siempre-negativo tiene accuracy=0.95 y recall=0.
- **Proxy SCS-CN:** `S = (25400/CN) - 254`, `Ia = 0.2*S`, `Q = (P-Ia)²/(P-Ia+S)` si `P > Ia`. CN=81.1 (ver `cuenca_la_honda_params.py`). P acumulado en rolling 24h. Aproximación de evento único aplicada a serie continua — documentar limitación; calibrar con nivel real cuando llegue de AMVA/SIATA.

## Visión computacional (Capa 2, OpenCV)

- Segmentación por color/contorno contra marcas de referencia fijas pintadas en el muro. **NO usar Roboflow/supervision** — evaluado y descartado, es toolkit de post-proceso, requiere detector/segmentador propio entrenado, overkill y frágil ante lluvia/turbidez/luz nocturna.
- Marcas de referencia: patrón **blanco/negro alternado** (no colores) — cámaras IP nocturnas pasan a monocromo con IR propio (850-940nm), colores distintos pueden verse igual en escala de grises. Cinta retroreflectante mejora captura nocturna. No requiere luz blanca adicional.
- Turbidez es sensor óptico dedicado (LED+fotodiodo), independiente de la cámara.

## Sensor ultrasónico (nivel)

`JSN-SR04T` (IP67, para intemperie) — **NO confundir con `HC-SR04`** (el del prototipo de calidad de aire, no aguanta lluvia). Montado fijo en muro/poste sobre el cauce a altura conocida; distancia medida ↓ = nivel de agua ↑.

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Hardware gateway | Raspberry Pi 4 4GB |
| Hardware nodos | ESP32 + SX1278 (LoRa) |
| Comunicación | LoRa 915 MHz |
| Visión computacional | Python + OpenCV |
| ML predictivo | Python + scikit-learn (RandomForestRegressor en experimentos; RandomForestClassifier en producción) |
| Broker MQTT | Mosquitto |
| Backend dashboard | Streamlit (dashboard/app.py) |
| Base de datos | SQLite (local) + PostgreSQL (cloud, futuro) |
| Alertas | WhatsApp Business API / Telegram Bot |
| Datos externos | API Socrata `datos.gov.co` (IDEAM) |
| Gestión de secretos | python-dotenv + `.env` (no trackeado) |

## Fuentes de datos IDEAM (Socrata, `www.datos.gov.co`, librería `sodapy`)

| Dataset ID | Nombre | Uso |
|---|---|---|
| `hp9r-jxuu` | Catálogo Nacional de Estaciones IDEAM | Metadata: código, nombre, categoría, estado, lat/lon, municipio |
| `s54a-sgyg` | Precipitación histórica | Serie larga para entrenamiento — schema confirmado con `codigoestacion`, `codigosensor`, `fechaobservacion`, `valorobservado`, `nombreestacion`, `departamento`, `municipio`, `zonahidrografica`, `latitud`, `longitud`, `descripcionsensor`, `unidadmedida` |
| `ksew-j3zj` | Precipitación cuasi tiempo real | Poca profundidad histórica, sirve para el gateway simulado en vivo |

Schema confirmado de `hp9r-jxuu` (columnas en minúscula vía sodapy): `codigo`, `nombre`, `categoria`, `tecnologia`, `estado`, `departamento`, `municipio`, `ubicacion` (o `latitud`/`longitud` planos), `altitud`, `fecha_instalacion`, `fecha_suspension`, `area_operativa`, `corriente`, `area_hidrografica`, `zona_hidrografica`, `subzona_hidrografica`, `entidad`.

Las credenciales Socrata se leen desde variables de entorno (`SODAPY_USERNAME`, `SODAPY_PASSWORD`) mediante `python-dotenv`. Ver `.env.example` en la raíz. Opcionalmente se puede usar `SODAPY_APP_TOKEN` para evitar rate limits estrictos; sacar token gratis en `dev.socrata.com` y pasarlo al cliente `Socrata(..., app_token=...)`. El archivo `.env` está en `.gitignore` y nunca debe subirse.

## Prototipo IoT ya operativo (base de código para nodos ESP32 de campo)

Estación calidad de aire v13, Wokwi: ESP32 + DHT22 (temp/humedad) + MQ-135 (CO2) + HC-SR04 (presencia) + OLED SSD1306 + LED RGB. Publica a Ubidots vía MQTT cada 90s. Servidor web local para config remota.

**Issues pendientes (aplican también al escalar a nodos CentineLA — mismo código base):**
- El sketch Wokwi no está versionado en este repositorio, por lo que sus credenciales WiFi expuestas no afectan directamente el repo. Si se reincorpora, debe moverse a `secrets.h` / `.env` y agregarse al `.gitignore`.
- `WiFiMulti`: agota 10 intentos → **offline permanente**, requiere reset manual — endurecer antes de llevar a campo (nodo de alerta no puede quedar mudo sin supervisión)
- Umbral `TEMP_CALOR = 28.0°C` dispara falso rojo en clima normal Medellín (confirmado con lectura real 28.30°C) — subir a 30-32°C o agregar histéresis
- Token Ubidots placeholder, config web no persiste a EEPROM, device label "prueba" — cosméticos, baja prioridad
- Las credenciales Socrata del script Python ya fueron movidas a `.env` y el historial de Git fue reescrito para eliminarlas.

## Estructura del repo (`CentineLA/`)

```
CentineLA/
  .env.example                   # nombres de variables de entorno, sin valores
  .gitignore                     # excluye .env, venv, datos crudos/procesados y modelos
  README.md                      # documentación principal del proyecto
  requirements.txt               # pandas, sodapy, scikit-learn, joblib, streamlit, numpy, matplotlib, plotly, folium, streamlit-folium, requests, python-dotenv
  scripts/
    01_buscar_estaciones.py      # filtra catálogo IDEAM por Valle de Aburrá, rankea por distancia a La Honda
    02_pull_historico.py         # pull histórico por estación (paginación + bisección recursiva + dedup); lee credenciales de .env
  data/
    raw/                         # NO trackeado: historico_*.csv, estaciones_candidatas.csv
    processed/                   # NO trackeado excepto .png: proxy, datasets, log_gateway
  simulate/
    cuenca_la_honda_params.py    # área/longitud/tramos/CN (dato-duro vs supuesto marcado)
    04_scs_cn_proxy.py           # proxy SCS-CN horario
    05_validar_proxy_eventos.py  # validación exploratoria sin inferir ground truth
    06_etl_features.py           # ETL features + Q_actual/Q_lags + split preservado
    07_entrenar_modelo_6h.py     # 3 enfoques RF regresor; joblib dump
    08_clasificador_6h.py        # clasificador binario + barrido de umbrales (probabilidad cruda)
    09_gateway_simulado.py       # gateway streaming con validación batch vs stream
    models/                      # trackeado: clf_6h.joblib, clf_6h_baseline_20260913.joblib; rf_6h*.joblib ignorado
  dashboard/
    app.py                       # Streamlit — Fase 3 CERRADA
    _check_fixes.py              # verificaciones del dashboard
    requirements.txt             # requirements pinnados del dashboard
  centinela-demo/
    CentineLA_Sesion_Fase3_Export.md  # resumen del cierre de Fase 3
```

## Seguridad y gestión de secretos

- Las credenciales de Socrata (`SODAPY_USERNAME`, `SODAPY_PASSWORD`) se leen desde un archivo `.env` mediante `python-dotenv`. Ver `.env.example`.
- `.env` está en `.gitignore` y **nunca** debe subirse al repositorio.
- El historial de Git fue reescrito con `git checkout --orphan` + force-push para eliminar las credenciales hardcodeadas que existían en versiones anteriores de `scripts/02_pull_historico.py`. Aun así, las credenciales expuestas deben considerarse comprometidas y rotarse.
- Para nodos ESP32/Wokwi futuros, usar `secrets.h` o equivalente y agregarlo a `.gitignore`.

## Convenciones de código en este proyecto

- Comentarios y docstrings en español
- Rutas relativas desde `scripts/` (`../data/raw/...`) — **siempre `os.makedirs(..., exist_ok=True)` antes de escribir**, no asumir que la carpeta existe (ya mordió una vez por carpetas vacías perdidas en un zip)
- Rutas de salida robustas: los scripts ya calculan `data/raw` desde la ubicación del archivo con `Path(__file__)`, para que funcionen igual desde `scripts/` o desde el root del proyecto
- `pull_dataset` en `02_pull_historico.py` pagina por fecha con tamaño fijo de página, loguea cuántas páginas/filas trajo, deduplica por `campo_codigo`, `fechaobservacion`, `codigosensor` y `valorobservado`, e imprime un resumen de `codigosensor`/`descripcionsensor` si hay más de un sensor por estación
- Nunca split aleatorio en datos de series de tiempo
- Scripts standalone ejecutables con `python nombre.py`, no notebooks
- Nombres de columna: siempre inspeccionar schema real antes de filtrar (Socrata cambia convenciones entre datasets) — no asumir nombres de campo a ciegas

## Roadmap — estado al 2026-07-11

1. **[CERRADO]** Pull histórico estaciones `0027015290`, `0027015310`, `0027015330`
2. **[CERRADO]** ETL features (lags, rolling, Q_actual/Q_lags, split cronológico)
3. **[CERRADO]** Proxy SCS-CN → `proxy_q_la_honda.csv`
4. **[CERRADO]** Modelado: 3 enfoques RF regresor + clasificador RF con barrido de umbrales
5. **[CERRADO]** Gateway simulado en modo streaming con validación batch vs stream
6. **[CERRADO — Fase 3]** Dashboard Streamlit consumiendo `log_gateway_simulado.csv` + `clf_6h.joblib`. Ver `dashboard/app.py`.
7. **[PENDIENTE CRÍTICO]** Derecho de petición formal a `atencionusuario@metropol.gov.co` — solo se envió aviso informal a `contacto@siata.gov.co`. Sin respuesta no hay fechas reales de eventos para validación honesta.
8. **[PENDIENTE FINAL]** Documento de 5pp + video entrega

---

## Fase 2 — Decisiones de diseño clave

- **Olaya Herrera sensor:** 240 canónico para el pipeline; 257 descartado del modelo, conservado como QA.
- **P_basin:** promedio `skipna=True` de las 3 estaciones por hora. Resample 1h con `sum()` (datos son incrementos de pluviómetro de balde cada ~10 min, no acumulados diarios).
- **SCS-CN:** CN=81.1 desde `cuenca_la_honda_params.py`. P acumulado como rolling 24h. Aplicación de fórmula de evento único a serie continua es aproximación — documentada como limitación; calibrar con nivel real cuando llegue.
- **Target de clasificación:** `label_6h = target_6h > p90(train)`. p90 calculado solo sobre train — nunca sobre test (fuga si no). Con distribución actual p90=0.0.
- **Umbral binario histórico del gateway:** 0.20. Recall=0.74, FP/TP≈1.5:1. El clasificador es una fuente dentro de la fusión multi-sensor (regla ≥2 fuentes críticas = rojo), no árbitro único — se acepta más FP a cambio de no perderse eventos.
- **Escala ternaria del dashboard (2026-10):** NORMAL `<0.30` / PRECAUCIÓN `[0.30, 0.70)` / ALERTA `≥0.70` sobre la misma `predict_proba`. Define el estado mostrado al operador y comunitario; no sustituye la regla multi-fuente.
- **Horizonte 6h priorizado:** tiempo de concentración corto + deadline oct-2026. Datasets 12h/24h generados pero no entrenados con el mismo rigor.

## Fase 2 — Resultado de modelado honesto

### Regresión RF (3 enfoques — todos fallaron vs baseline)

| Enfoque | MAE global | R² global | Eventos MAE | vs Baseline |
|---|---|---|---|---|
| RF target absoluto | 0.036479 | 0.221064 | 0.350105 | peor |
| RF delta residual | 0.038530 | 0.149419 | 0.370445 | peor |
| RF weighted (×20) | 0.041974 | -0.083693 | 0.374840 | peor |
| **Baseline persistencia** | **0.014288** | **0.510524** | **0.189361** | referencia |

Causa: 0.85% de eventos en el dataset. RF diluyeresiduals hacia la media estructuralmente.

### Clasificador RF (enfoque final)

**Barrido de umbrales — corrida del 27-jul-2026 (referencia histórica, dataset de 43,068 filas):**

| Umbral | Recall | Precision | FN | FP | TP |
|---|---|---|---|---|---|
| 0.50 | 0.607 | 0.612 | 144 | 141 | 222 |
| 0.40 | 0.658 | 0.539 | 125 | 206 | 241 |
| 0.30 | 0.702 | 0.477 | 109 | 282 | 257 |
| **0.20** | **0.743** | **0.400** | **94** | **408** | **272** |
| 0.15 | 0.781 | 0.311 | 80 | 633 | 286 |
| 0.10 | 0.817 | 0.257 | 67 | 865 | 299 |

**Métricas con umbrales ternarios (corrida del 03-oct-2026):**

| Umbral | Recall    | Precisión | Rol en el sistema                |
|--------|-----------|-----------|----------------------------------|
| 0.20   | 0.752998  | 0.415344  | referencia binaria anterior      |
| 0.30   | 0.714628  | 0.497496  | PRECAUCIÓN (amarillo) o más      |
| 0.70   | 0.549161  | 0.860902  | ALERTA (rojo)                    |

**Parámetros:** Train 35,315 filas (hasta 2025-07-06 15:00), Test 9,051 filas (2025-07-06 16:00 → 2026-09-30), 417 positivos en test. Reproducible con `python simulate/10_metricas_ternario.py`.

**Lectura:** el amarillo captura más crecientes (71%) pero la mitad de sus avisos son falsas alarmas; el rojo detecta menos (55%) pero 86% de sus avisos son reales.

Techo real de recall aislado: ~0.82. Limitación mitigada por fusión multi-fuente en la arquitectura.

## Fase 2 — Gateway simulado

- `simulate/09_gateway_simulado.py` valida features streaming vs batch con 500 timestamps aleatorios, epsilon 1e-6 — **validación pasó limpia**.
- Inferencia sobre serie completa (~9.5 años): 4947 alertas ALERTA_6H / 38918 NORMAL (log binario del gateway). El dashboard lo re-etiqueta a NORMAL / PRECAUCIÓN / ALERTA con los umbrales ternarios.
- Ventana de test real (2025-07-06 → 2026-07-09): 693 alertas en 7883 filas (8.8%), consistente con barrido de umbrales (~680 esperado para umbral 0.20) **(corrida previa, al 13-sep-2026)**.
- Ventana de test actual (2025-07-06 → 2026-09-30): 9,051 filas, 417 positivos (label_6h=1).
- Implementación: micro-lotes de 2048 filas con `n_jobs=1` en inferencia para evitar overhead de threading.

## Lecciones técnicas — Fase 2

1. **Validar eventos contra la propia serie es circular.** Cualquier fecha de "evento" inferida desde máximos locales del proxy contamina la validación. Requiere fecha externa confirmada (PQRSD AMVA/DAGRD).
2. **Filtro de calidad puede ocultar huecos de instrumento no documentados.** Las 36k filas descartadas revelaron que el hueco de Metromedellín empieza en feb-2019, no jul-2019 como se estimó en Fase 1. Cruzar siempre contra CSV crudo antes de asumir artefacto de pipeline.
3. **Probar el baseline ingenuo antes de entrenar.** Un baseline de persistencia superó a los 3 enfoques de RF en regresión — si se hace al inicio, ahorra iteraciones de tuning.
4. **Reformular el problema puede ser más efectivo que tunear hiperparámetros.** El pivote a clasificación binaria (semáforo) fue más productivo que iterar sobre el mismo target continuo.
5. **Accuracy es inútil con clases muy desbalanceadas.** Baseline siempre-negativo: accuracy=0.95, recall=0. La métrica real para alerta es recall de clase positiva.
6. **Inferencia fila-a-fila con RF en Python puro genera overhead masivo.** Usar micro-lotes (≥1000 filas) y `n_jobs=1` en producción; `n_jobs=-1` solo para entrenamiento.