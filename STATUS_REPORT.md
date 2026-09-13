# STATUS_REPORT — Auditoría CentineLA

Fecha de auditoría: 2026-07-27  
Modo: auditoría posterior a limpieza de credenciales y reescritura de historial.
Nota: el directorio **sí es un repositorio Git**. Tras la reescritura de historial, la rama `master` tiene un historial limpio con los archivos actuales; los commits anteriores que contenían credenciales hardcodeadas fueron eliminados localmente y en `origin` mediante force-push. Las credenciales afectadas aún deben considerarse comprometidas y rotarse.

---

## 1. Pipeline de datos (Fase 1)

- ✅ **Scripts de extracción/limpieza IDEAM encontrados**  
  - [scripts/01_buscar_estaciones.py](scripts/01_buscar_estaciones.py) — mtime 2026-07-09 22:46:09  
  - [scripts/02_pull_historico.py](scripts/02_pull_historico.py) — mtime 2026-07-10 21:21:43  
  - [simulate/04_scs_cn_proxy.py](simulate/04_scs_cn_proxy.py) — mtime 2026-07-11 11:44:17  
  Las 3 estaciones objetivo están referenciadas: `0027015290` (Pajarito), `0027015310` (Metromedellín), `0027015330` (Olaya Herrera) — sensor 240 canónico, 257 como QA ([simulate/04_scs_cn_proxy.py](simulate/04_scs_cn_proxy.py#L48)).

- ✅ **Las fechas de hueco de Metromedellín en `06_etl_features.py` fueron corregidas**
  - El contexto técnico documenta los huecos reales: Metromedellín `2019-02-18 16:00` → `2019-07-26 11:00`; Olaya 240 `marzo-jun 2019` → retoma `2019-08-24` ([CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L12)).  
  - El código actual usa las fechas correctas en [simulate/06_etl_features.py](simulate/06_etl_features.py#L32-L35):
    ```python
    METROMEDELLIN_GAP_INICIO = pd.Timestamp("2019-02-18 16:00:00")
    METROMEDELLIN_GAP_FIN    = pd.Timestamp("2019-07-26 11:00:00")
    ```
  - Estas constantes se usan solo para **diagnóstico** de meses con baja disponibilidad; el filtro real sigue siendo `n_estaciones_disponibles < 2`, como se documenta.

- ✅ **CN compuesto = 81.1**  
  - [simulate/cuenca_la_honda_params.py](simulate/cuenca_la_honda_params.py#L72): `CN = calcular_cn_compuesto()`  
  - Verificación en runtime: `from cuenca_la_honda_params import CN` devuelve `81.1`.

- ✅ **Filtro `n_estaciones ≥ 2` aplicado antes de features**  
  - [simulate/06_etl_features.py](simulate/06_etl_features.py#L29): `LOW_QUALITY_MIN_ESTACIONES = 2`  
  - [simulate/06_etl_features.py](simulate/06_etl_features.py#L148-L149): `filas_calidad = df[df["n_estaciones_disponibles"] >= LOW_QUALITY_MIN_ESTACIONES].copy()`  
  - [simulate/09_gateway_simulado.py](simulate/09_gateway_simulado.py#L38): `LOW_QUALITY_MIN_ESTACIONES = 2`  
  - [simulate/09_gateway_simulado.py](simulate/09_gateway_simulado.py#L67): `proxy = proxy[proxy["n_estaciones_disponibles"] >= LOW_QUALITY_MIN_ESTACIONES].copy()`

---

## 2. Modelo ML (Fase 2)

- ✅ **Script de entrenamiento del RandomForestClassifier encontrado**  
  - [simulate/08_clasificador_6h.py](simulate/08_clasificador_6h.py) — mtime 2026-07-11 13:43:29  
  - [simulate/07_entrenar_modelo_6h.py](simulate/07_entrenar_modelo_6h.py) — mtime 2026-07-11 13:29:22 (contiene los 3 enfoques de regresión previos).

- ✅ **class_weight usado = `"balanced"`**  
  - [simulate/08_clasificador_6h.py](simulate/08_clasificador_6h.py#L153): `class_weight="balanced"`

- ✅ **Umbral de decisión operativo = 0.20 (NO 0.5)**  
  - [simulate/08_clasificador_6h.py](simulate/08_clasificador_6h.py#L223) incluye `0.20` en el barrido de umbrales.  
  - [simulate/09_gateway_simulado.py](simulate/09_gateway_simulado.py#L33): `UMBRAL_ALERTA = 0.20`  
  - [dashboard/app.py](dashboard/app.py#L31): `UMBRAL_ALERTA = 0.20`

- ✅ **Recall y precisión reportados en el output más reciente (ejecución 2026-07-21)**  
  - Ejecución de `python simulate/08_clasificador_6h.py`:  
    - Umbral `0.20` → **recall = 0.743169**, **precision = 0.400000**, FN=94, FP=408, TP=272.  
  - Esto coincide con lo documentado en [CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L154-L155).

- ✅ **Rastro de los 3 intentos de RandomForestRegressor**  
  - Código vivo en [simulate/07_entrenar_modelo_6h.py](simulate/07_entrenar_modelo_6h.py):  
    - Target absoluto: líneas ~113-150  
    - Weighted (`sample_weight`): líneas ~152-195  
    - Delta residual: líneas ~197-240  
  - Modelos huérfanos en disco:  
    - `simulate/models/rf_6h.joblib`  
    - `simulate/models/rf_6h_delta.joblib`  
    - `simulate/models/rf_6h_weighted.joblib`

- ✅ **Validación de leakage temporal en gateway simulado pasa**  
  - Ejecución de `python simulate/09_gateway_simulado.py` (2026-07-21 19:50):  
    - `VALIDACION OK: 500 timestamps aleatorios comparados, sin diferencias > 1e-06.`  
  - [simulate/09_gateway_simulado.py](simulate/09_gateway_simulado.py#L39): `EPSILON = 1e-6`  
  - Output: 43865 filas inferidas, 4947 alertas / 38918 normal.

- ✅ **No hay referencia a SMOTENC en el código**  
  - Búsqueda `SMOTENC|smotenc|SMOTE` en `.py` y `.md` no arrojó coincidencias.

---

## 3. Dashboard Streamlit (Fase 3)

- ✅ **Dashboard principal encontrado y Fase 3 cerrada**  
  - [dashboard/app.py](dashboard/app.py) — implementa Vista Pública y Vista Operador/JAC.  
  - [dashboard/_check_fixes.py](dashboard/_check_fixes.py) — verificaciones de fixes y simulador.
  - Se creó un `README.md` completo en la raíz del repositorio con la documentación técnica principal.

- ✅ **Existen las dos vistas**  
  - Vista Pública: [dashboard/app.py](dashboard/app.py#L407) — semáforo, card probabilidad, 3 cards acumulado lluvia, mapa Folium con fallback offline, calendario 90 días.  
  - Vista Operador/JAC: [dashboard/app.py](dashboard/app.py#L541) — log crudo, gráfico histórico, métricas modelo, feature importances, inspector histórico, simulador interactivo.

- ⚠️ **Preset "lluvia fuerte" del simulador no arroja ~77%**  
  - Replicación exacta del preset en [dashboard/app.py](dashboard/app.py#L613-L617) con `hora_dia=14.0, mes=7.0` (como en `_check_fixes.py`):  
    - **Probabilidad real = 0.9650 (96.50%)**  
  - Con hora del sistema actual: **98.00%**.  
  - Valor esperado ~77% **no coincide**. El modelo dispara ALERTA_6H (umbral 0.20), pero con probabilidad mucho más alta.

- ❌ **`inject_design_system()` NO existe**  
  - Búsqueda en todo el repo no arrojó la función.  
  - El dashboard aplica un bloque CSS inline básico (`CSS_GLOBAL` en [dashboard/app.py](dashboard/app.py#L673-L691)), no hay tokens de color, tipografía Space Grotesk/Inter ni no-line cards.  
  - No hay evidencia de aplicación del "Luminous Engine" design system.

- ✅ **Fallback offline del mapa existe y no depende de internet**  
  - [dashboard/app.py](dashboard/app.py#L264) `check_red()` detecta conectividad a OSM.  
  - [dashboard/app.py](dashboard/app.py#L289-L290) `mostrar_mapa()` cae a `_mapa_estatico()` si no hay red.  
  - `_mapa_estatico` usa `matplotlib.pyplot` y genera PNG en memoria — no requiere tiles externos.

---

## 4. Documento técnico (5 páginas) y video

- ❌ **Documento técnico de 5 páginas no encontrado**  
  - Búsquedas por `*informe*`, `*documento*`, `*.docx`, `*.pdf` no arrojaron resultados.  
  - [CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md) es **contexto interno para el asistente**, no el documento de entrega.  
  - [centinela-demo/CentineLA_Sesion_Fase3_Export.md](centinela-demo/CentineLA_Sesion_Fase3_Export.md) marca Fase 3 como cerrada, pero no contiene un documento formal de 5 páginas.

- ❌ **Video / guion / storyboard no encontrados**  
  - Búsquedas por `*video*`, `*storyboard*`, `*guion*`, extensiones de video (`mp4`, `mov`, `avi`) no arrojaron resultados.

- ❌ **Render conceptual (Gemini/poste instalado) no encontrado**  
  - No hay archivos con `*render*`, `*.blend`, `*.glb`, `*.gltf`, `*.fbx`, ni imágenes de render conceptual.  
  - No es posible verificar si las 3 correcciones (sensores ultrasónico/turbidez altos, cámara apuntando a la mira, altavoz fuera del BOM) fueron aplicadas.

---

## 5. PQRSD

- ❌ **Copia del PDF del derecho de petición no encontrada en el repo**  
  - No hay carpeta `docs/`. Búsquedas por `*derecho*`, `*peticion*`, `*.pdf`, `PQRSD` no arrojaron resultados.

- ❌ **No hay documento real para citar la dirección de correo**  
  - El contexto técnico menciona `atencionusuario@metropol.gov.co` ([CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L192)), pero también advierte que solo se envió un aviso informal a `contacto@siata.gov.co`.  
  - No se encontró PDF/norra que contenga `atencionalciudadano@metropol.gov.co` para resolver la discrepancia.

- ❌ **No hay respuesta de SIATA recibida en el repo**  
  - No hay emails, PDFs adjuntos ni notas que registren contenido/fecha de respuesta.

---

## 6. Estado general del repo

- ❌ **No hay historial de commits Git**  
  - `git log --oneline -10` falla: `fatal: not a git repository`.  
  - No es posible reportar últimos 10 commits.

- ⚠️ **TODOs / pendientes / comentarios de revisión encontrados**
  - [CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L130): "Issues pendientes" del prototipo IoT (credenciales WiFi expuestas, `WiFiMulti`, umbral `TEMP_CALOR`).
  - [CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L192): "PENDIENTE CRÍTICO — Derecho de petición formal a `atencionusuario@metropol.gov.co`".
  - [CENTINELA_CONTEXTO_TECNICO.md](CENTINELA_CONTEXTO_TECNICO.md#L193): "PENDIENTE FINAL — Documento de 5pp + video entrega".
  - [simulate/05_validar_proxy_eventos.py](simulate/05_validar_proxy_eventos.py#L26): "Fechas exactas pendientes — ver PQRSD a `atencionusuario@metropol.gov.co` o fuente DAGRD directa.".
  - [simulate/cuenca_la_honda_params.py](simulate/cuenca_la_honda_params.py#L60): nota sobre reparto de área y CN como supuestos sin fuente oficial.
  - [centinela-demo/CentineLA_Sesion_Fase3_Export.md](centinela-demo/CentineLA_Sesion_Fase3_Export.md#L58): "Pulido estético adicional del mapa y responsividad en pantallas angostas — no verificado a fondo".

- ✅ **Dependencias declaradas vs realmente importadas**
  - [requirements.txt](requirements.txt) raíz ahora declara: `pandas`, `sodapy`, `scikit-learn`, `joblib`, `streamlit`, `numpy`, `matplotlib`, `plotly`, `folium`, `streamlit-folium`, `requests`, `python-dotenv`.
  - El archivo raíz es suficiente para ejecutar tanto el pipeline completo como el dashboard.
  - [dashboard/requirements.txt](dashboard/requirements.txt) sigue disponible con versiones pinnadas para reproducibilidad exacta del dashboard.

---

## 7. Seguridad y credenciales

- ✅ **Credenciales Socrata movidas a variables de entorno**
  - `scripts/02_pull_historico.py` ya no contiene `username`/`password` hardcodeados.
  - Lee `SODAPY_USERNAME` y `SODAPY_PASSWORD` desde el entorno mediante `python-dotenv`.
  - Se creó `.env.example` con los nombres de variable (sin valores).
  - Se agregó `.env` a `.gitignore`.
  - Se agregó `python-dotenv` a `requirements.txt`.

- ✅ **Historial de Git reescrito**
  - Se ejecutó `git checkout --orphan clean-main`, `git gc --prune=now --aggressive` y `git push -f origin master` para eliminar los commits que contenían las credenciales.
  - Commit actual del historial limpio: `CentineLA: historial limpio`.

- ⚠️ **Rotación de credenciales aún pendiente**
  - Las credenciales estuvieron expuestas en GitHub; aunque ya no están en el historial actual, deben rotarse en Socrata/datos.gov.co.
  - El archivo `.env` (no trackeado) debe completarse localmente con las nuevas credenciales.

---

## Resumen ejecutivo

| Área | Estado real vs asumido |
|---|---|
| Fase 1 pipeline | ✅ Scripts completos; CN=81.1; filtro n≥2 presente. ❌ Filtro de huecos exactos no codificado. |
| Fase 2 ML | ✅ Clasificador RF con `class_weight="balanced"`, umbral 0.20, recall 0.743/precision 0.400. ✅ 3 intentos de RF regresor rastreables. ✅ Gateway valida. ✅ Sin SMOTENC. |
| Fase 3 Dashboard | ✅ Dos vistas y fallback offline funcionan. ⚠️ Preset lluvia fuerte da 96.5% (no ~77%). ❌ `inject_design_system()` no existe. |
| Documento/video | ❌ No iniciados/encontrados. |
| PQRSD | ❌ No hay copia del derecho de petición ni respuesta. |
| Seguridad | ✅ Credenciales movidas a `.env`. ✅ Historial reescrito. ⚠️ Credenciales expuestas deben rotarse. |
| Repo general | ✅ Es Git (historial limpio en `master`). ✅ `requirements.txt` raíz completo. |
