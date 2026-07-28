# CentineLA - Sesion Fase 3 (Dashboard) - Export

Fecha: 2026-07-12
Estado: Fase 3 cerrada (empaquetado y verificacion final)

## 1) Estructura del dashboard

El dashboard queda en dos vistas:

- Vista Publica
  - Semaforo de estado actual (NORMAL / ALERTA_6H) con probabilidad asociada.
  - Tarjetas de lluvia acumulada (ultimas 24h, ultimos 30 dias, anio en curso).
  - Mapa de nodos (interactivo cuando hay internet, fallback estatico cuando no hay internet).
  - Calendario de alertas de los ultimos 90 dias (enfoque visual simple para publico no tecnico).

- Vista Operador (JAC)
  - Tabla del log de gateway (ultimas filas).
  - Serie historica de lluvia y proxy de escorrentia con marcas de alerta.
  - Metricas del clasificador a umbral operativo (0.20).
  - Importancia de variables del modelo.
  - Inspeccion de punto historico por fecha/hora.
  - Simulador interactivo del modelo con presets y boton Calcular prediccion.

## 2) Decisiones de diseno

- Mapa con fallback offline
  - Decision: mantener dos modos de render (folium si hay internet; imagen/figura estatica si no).
  - Motivo: asegurar continuidad de demo sin dependencia de conectividad externa.

- Calendario de alertas en Vista Publica (en lugar de grafico dual-axis)
  - Decision: usar una visualizacion de dias en rojo/verde para eventos.
  - Motivo: lectura inmediata para publico no tecnico; reduce sobrecarga cognitiva frente a ejes duales.

- Simulador interactivo con presets
  - Decision: incluir escenarios rapidos (lluvia fuerte, aguacero puntual, normal) y controles manuales.
  - Motivo: permite demo en vivo del comportamiento del modelo, incluso si el dia de sustentacion no presenta evento real en el historico.

## 3) Bugs encontrados y resueltos

- Plotly titlefont (API vieja)
  - Sintoma: error por uso de propiedad deprecada.
  - Causa: cambio de API en Plotly.
  - Solucion: migracion a title con font anidado.

- Zigzag en interpolacion de nodos del mapa
  - Sintoma: trazo visual poco realista entre anclas.
  - Causa: interpolar lat/lon de forma independiente contra anclas no colineales.
  - Solucion: interpolacion por segmentos respetando anclas reales del recorrido.

- Marcador de ancla desalineado
  - Sintoma: el simbolo parecia corrido respecto al punto esperado.
  - Causa: diferencia entre centro visual de bounding-box triangular y centro geometrico de circulo.
  - Solucion: ajuste de marcador/etiqueta para alinear centro real del nodo.

- Tarjetas de lluvia en 0.0
  - Sintoma: sospecha de bug en acumulados recientes.
  - Diagnostico: dato real del periodo, no falla de calculo.
  - Verificacion: contraste con serie en data/processed/proxy_q_la_honda.csv.

## 4) Pendiente no bloqueante

- Pulido estetico adicional del mapa y responsividad en pantallas angostas.
- Estado: no verificado a fondo; no bloquea demo de prototipo.

## 5) Instrucciones de ejecucion para la sustentacion

1. Activar el entorno virtual del proyecto.
2. Asegurarse de que existe un archivo `.env` en la raíz con las credenciales Socrata (`SODAPY_USERNAME`, `SODAPY_PASSWORD`). Ver `.env.example`.
3. Ejecutar: streamlit run dashboard/app.py
4. Verificar carga en navegador local (http://localhost:8501).
5. Para demo offline, apagar WiFi:
   - El mapa debe caer automaticamente al fallback estatico.
   - El resto del dashboard debe mantenerse operativo.

Nota: el pipeline de descarga IDEAM (`scripts/02_pull_historico.py`) requiere las credenciales en `.env`. El dashboard solo necesita los archivos ya generados en `data/processed/`.

## 6) Cierre de fase

Se cierra Fase 3 como completada para demo funcional.

Actualizaciones posteriores al cierre de esta sesión:
- Se creó un `README.md` completo en la raíz del repositorio.
- Se creó `.env.example` y se movieron las credenciales de `scripts/02_pull_historico.py` a variables de entorno (`SODAPY_USERNAME`, `SODAPY_PASSWORD`) usando `python-dotenv`.
- Se agregó `.env` a `.gitignore` y se reescribió el historial de Git para eliminar las credenciales hardcodeadas.

No se realizan mas cambios de codigo en dashboard/app.py salvo falla critica detectada en smoke test.