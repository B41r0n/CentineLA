"""
Parámetros morfométricos e hidrológicos de la microcuenca Quebrada La Honda,
para el proxy lluvia-escorrentía (método SCS-CN) de Fase 2.

FUENTES:
- Área, longitud, límites, tramos: AMVA, ficha técnica oficial
  https://www.metropol.gov.co/noticias/quebrada-la-honda
- Uso del suelo por tramo: AMVA, PMI Microcuenca La Honda (Plan Quebradas)
  https://www.metropol.gov.co/ambiental/recurso-hidrico/Informacionrecursohidrico/PMI-Microcuencas/08-La-Honda/PMI_LA_HONDA%20.pdf
- Pendientes: compartidas con microcuenca El Molino (mismo sistema de ladera
  nororiental, La Honda es tributaria directa de El Molino)
  https://es.wikipedia.org/wiki/Quebrada_El_Molino

DATO DURO (oficial AMVA) vs SUPUESTO (mío, sin fuente oficial) — marcado
explícitamente en cada constante. No mezclar sin revisar.
"""

# ── DATO DURO (AMVA) ─────────────────────────────────────────────────────

AREA_TOTAL_KM2 = 5.9          # área de la microcuenca completa
LONGITUD_CAUCE_KM = 6.2       # longitud del cauce principal

LIMITES = {
    "norte": "microcuenca El Molino",
    "sur": "microcuenca Santa Elena",
    "oriente": "microcuenca Piedras Blancas",
    "occidente": "río Aburrá-Medellín",
}

TRAMOS = {
    "alto": {
        "descripcion": "nacimiento hasta límite suelo rural/urbano",
        "ubicacion": "corregimiento Santa Elena, Parque Ecoturístico Arví",
        "cobertura": "poco intervenido, buena cobertura vegetal (bosque)",
        "pendiente_pct": (30, 999),  # DATO DURO (compartido con El Molino): >30%
        "riesgo": "alta susceptibilidad a movimientos en masa (flujos de escombro, caída de roca)",
    },
    "medio": {
        "descripcion": "inicio barrio La Honda hasta carrera 45",
        "ubicacion": "barrios La Honda, La Hondita, Versalles 1 y 2, El Raizal",
        "cobertura": "mayor densidad poblacional, urbanización sin planificación",
        "pendiente_pct": (12, 25),  # DATO DURO (compartido con El Molino)
        "riesgo": "AMENAZA ALTA por avenidas torrenciales — este es el tramo objetivo de CentineLA",
    },
    "bajo": {
        "descripcion": "carrera 45 hasta desembocadura en río Aburrá-Medellín",
        "ubicacion": "Jardín Botánico, Brasilia, Miranda, Moravia",
        "cobertura": "urbano, corredores viales, corredor biológico Jardín Botánico/San Pedro",
        "pendiente_pct": (6, 12),  # DATO DURO (compartido con El Molino)
        "riesgo": "socavación del cauce principal",
    },
}

# ── SUPUESTO (mío — sin fuente oficial, ajustar si aparece dato real) ─────

# Reparto de área por tramo: NO hay dato publicado de cuántos km² es cada
# tramo. Este split es una suposición de trabajo basada en la descripción
# cualitativa (tramo alto = cabecera rural extensa en Parque Arví, tramos
# medio/bajo = franja urbana angosta siguiendo el cauce). AJUSTAR si se
# consigue delimitación real (SIATA, GIS, o fotointerpretación satelital).
SUPUESTO_REPARTO_AREA = {
    "alto": 0.40,
    "medio": 0.30,
    "bajo": 0.30,
}

# Curve Number (SCS, condición de humedad antecedente II) por tramo.
# Asume Hydrologic Soil Group C (suelos residuales/saprolíticos derivados
# de dunita típicos de la ladera nororiental de Medellín — moderada a baja
# infiltración). El grupo hidrológico real NO está confirmado con estudio
# de suelos (IGAC/IDEAM) — es la suposición estándar más razonable para la
# zona, no un dato medido.
SUPUESTO_CN_POR_TRAMO = {
    "alto": 70,   # bosque en buena condición, HSG C
    "medio": 92,  # residencial denso informal, lotes pequeños, ~65%+ impermeable, HSG C
    "bajo": 85,   # urbano mixto (vías + construcciones + corredor verde Jardín Botánico), HSG C
}


def calcular_cn_compuesto():
    """
    CN ponderado por área, usando SUPUESTO_REPARTO_AREA y SUPUESTO_CN_POR_TRAMO.
    Devuelve el CN compuesto para usar en la fórmula SCS-CN del proxy.
    ADVERTENCIA: depende de los dos SUPUESTOS de arriba, no es dato oficial.
    """
    cn = sum(
        SUPUESTO_REPARTO_AREA[tramo] * SUPUESTO_CN_POR_TRAMO[tramo]
        for tramo in TRAMOS
    )
    return round(cn, 1)


CN = calcular_cn_compuesto()


if __name__ == "__main__":
    print(f"Área total: {AREA_TOTAL_KM2} km²")
    print(f"Longitud cauce: {LONGITUD_CAUCE_KM} km")
    print(f"CN compuesto (SUPUESTO): {CN}")
    print()
    print("Fórmula SCS-CN lista para usar en Fase 2:")
    print("  S = (25400 / CN) - 254")
    print("  Ia = 0.2 * S")
    print("  Q = (P - Ia)^2 / (P - Ia + S)   si P > Ia, si no Q = 0")
