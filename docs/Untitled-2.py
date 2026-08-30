# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable,
    ListItem, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

DOC_TITLE = "Revisión de la propuesta VIE 2027 contra la Rúbrica de Evaluación"
SUBTITLE = "Laboratorio de Plasmas — Desarrollo y evaluación comparativa de reactores corona y DBD para PAW"

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleCustom", parent=styles["Title"], fontSize=17, leading=21,
    textColor=colors.HexColor("#1F3864"), spaceAfter=4
))
styles.add(ParagraphStyle(
    name="SubtitleCustom", parent=styles["Normal"], fontSize=10.5, leading=14,
    textColor=colors.HexColor("#44546A"), spaceAfter=14, alignment=TA_LEFT
))
styles.add(ParagraphStyle(
    name="SectionHeader", parent=styles["Heading1"], fontSize=13, leading=16,
    textColor=colors.white, backColor=colors.HexColor("#1F3864"),
    spaceBefore=14, spaceAfter=8, leftIndent=6, borderPadding=(4, 4, 4, 4)
))
styles.add(ParagraphStyle(
    name="SubHeader", parent=styles["Heading2"], fontSize=10.8, leading=13,
    textColor=colors.HexColor("#1F3864"), spaceBefore=8, spaceAfter=3
))
styles.add(ParagraphStyle(
    name="BodyJust", parent=styles["Normal"], fontSize=9.6, leading=13.2,
    alignment=TA_JUSTIFY, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="BulletJust", parent=styles["Normal"], fontSize=9.6, leading=13,
    alignment=TA_JUSTIFY
))
styles.add(ParagraphStyle(
    name="Alert", parent=styles["Normal"], fontSize=9.8, leading=13.5,
    alignment=TA_JUSTIFY, textColor=colors.HexColor("#7A1F1F")
))
styles.add(ParagraphStyle(
    name="TocNote", parent=styles["Normal"], fontSize=9, leading=12,
    textColor=colors.HexColor("#555555"), alignment=TA_LEFT
))

def bullets(items, style="BulletJust", bullet_color="#1F3864"):
    flow = []
    for it in items:
        flow.append(ListItem(Paragraph(it, styles[style]), leftIndent=12,
                              bulletColor=colors.HexColor(bullet_color)))
    return ListFlowable(flow, bulletType="bullet", start="•", leftIndent=14, spaceBefore=2, spaceAfter=8)

def section_header(text):
    return Paragraph(text, styles["SectionHeader"])

def subheader(text):
    return Paragraph(text, styles["SubHeader"])

styles.add(ParagraphStyle(
    name="CellHeader", parent=styles["Normal"], fontSize=8.6, leading=10.5,
    textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_LEFT
))
styles.add(ParagraphStyle(
    name="CellBody", parent=styles["Normal"], fontSize=8.3, leading=10.6,
    alignment=TA_JUSTIFY
))
styles.add(ParagraphStyle(
    name="CellBodyBold", parent=styles["Normal"], fontSize=8.3, leading=10.6,
    alignment=TA_LEFT, fontName="Helvetica-Bold"
))

def obligatorio_table(rows):
    """rows: list of (criterio, estado, accion) as plain strings"""
    header = [
        Paragraph("Requisito obligatorio (⛔)", styles["CellHeader"]),
        Paragraph("Estado actual", styles["CellHeader"]),
        Paragraph("Qué hay que hacer", styles["CellHeader"]),
    ]
    data = [header]
    for criterio, estado, accion in rows:
        data.append([
            Paragraph(criterio, styles["CellBodyBold"]),
            Paragraph(estado, styles["CellBody"]),
            Paragraph(accion, styles["CellBody"]),
        ])
    # Ancho disponible en letter con márgenes de 1.8cm a cada lado: 21.59 - 3.6 = ~17.99cm
    t = Table(data, colWidths=[4.6*cm, 2.6*cm, 8.8*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C00000")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBEAEA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t

story = []

# ---------- Portada / encabezado ----------
story.append(Paragraph(DOC_TITLE, styles["TitleCustom"]))
story.append(Paragraph(SUBTITLE, styles["SubtitleCustom"]))
story.append(Paragraph(
    "Convocatoria interna VIE 2027 — Vicerrectoría de Investigación y Extensión, "
    "Instituto Tecnológico de Costa Rica. Documento evaluado contra la "
    "<i>Rúbrica de Evaluación – Investigación Básica y Aplicada</i>.",
    styles["BodyJust"]
))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1F3864"), spaceBefore=6, spaceAfter=12))

# ---------- Resumen ejecutivo / alerta obligatorios ----------
story.append(section_header("Resumen ejecutivo: requisitos obligatorios (⛔) en riesgo"))
story.append(Paragraph(
    "La rúbrica establece que cuando un requisito marcado como obligatorio obtiene una calificación "
    "\"deficiente\" o \"regular\", el Comité Técnico emite <b>dictamen técnico desfavorable</b>, "
    "independientemente del puntaje total obtenido. Actualmente, varios de estos requisitos no se "
    "cumplen o no se pueden verificar con la información entregada. Se recomienda resolver esta tabla "
    "<b>antes</b> de atender el resto de observaciones, ya que cualquiera de estos puntos puede hacer que "
    "la propuesta no sea recomendada sin importar el resto de la calificación.",
    styles["BodyJust"]
))
story.append(Spacer(1, 6))

obligatorios_rows = [
    ["Participación estudiantil (Sección 1)",
     "No cumple",
     "No hay estudiantes asistentes ni TFG vinculado en los Cuadros 2 y 3. Incorporar al menos un/a estudiante de grado como asistente o vincular un Trabajo Final de Graduación."],
    ["Revisión previa de literatura / vacío de conocimiento (Sección 2)",
     "No verificable",
     "El estado del arte tiene párrafos incompletos y el propio equipo señala en comentarios internos que no queda claro el aporte original frente al antecedente más cercano [11]. Debe quedar explícito y no plantearse como actividad futura."],
    ["Enfoque, alcance y diseño metodológico (Sección 3)",
     "Desorganizado",
     "El apartado 4.1 dice \"AGREGAR\", pero la metodología de microalgas sí existe más adelante en el documento, fuera de lugar. Reorganizar el texto y eliminar el marcador \"AGREGAR\" para que la sección quede completa y coherente."],
    ["Población de estudio, criterios y tamaño muestral (Sección 3)",
     "No cumple",
     "No se define número de repeticiones/corridas para el CCD, número de muestras de PAW seguidas en el ensayo de estabilidad, ni número de plantas/réplicas por tratamiento en lechugas y microalgas."],
    ["Giras, trabajo de campo, pasantías o congresos (Sección 4)",
     "No verificable",
     "Depende del archivo Excel de Plan de Acción, no incluido. Verificar que los traslados al CIB para mediciones de nitritos/nitratos queden planificados con cronograma y presupuesto."],
    ["Responsable, apoyo y supervisor por actividad (Sección 4)",
     "No verificable",
     "Mismo archivo Excel no incluido. Confirmar que cada actividad tenga persona ejecutora, de apoyo (si aplica) y supervisora claramente diferenciadas."],
    ["Resultados esperados y plazos por actividad (Sección 4)",
     "No verificable",
     "Verificar en el Excel que cada actividad tenga un resultado esperado y un plazo, distinto de los productos académicos comprometidos."],
    ["Presupuesto dentro de los montos máximos (Sección 5)",
     "No verificable",
     "El Cuadro 4 está vacío y el desglose detallado vive en un Excel adicional no incluido. Completar ambos para poder verificar que se respetan los topes de la convocatoria."],
    ["Equipamiento e inversión concentrados en el primer año (Sección 5)",
     "No cumple",
     "Solo aparece un enlace suelto a un barotermohigrómetro sin monto ni justificación formal. Formalizar lista de equipo, costo y justificación, y confirmar que se concentre en el año 1."],
    ["Riesgo asociado a una mitigación por cada actividad (Sección 6)",
     "No verificable",
     "Depende del Excel no incluido. Confirmar que cada actividad tenga asociado al menos un riesgo con su respectiva acción de mitigación."],
    ["Viabilidad de las acciones de mitigación (Sección 6)",
     "No verificable",
     "Es una pregunta obligatoria distinta de la anterior: confirmar que cada acción de mitigación propuesta sea coherente, viable y suficiente para salvaguardar el éxito de la actividad, no solo que exista una acción declarada."],
    ["Declaración y justificación de aval ético o regulatorio (Sección 7 / Sección 14 del formulario)",
     "No cumple",
     "Ninguna de las 4 preguntas (a–d) está marcada y la declaración de exención dice \"Indique aquí las razones\" sin completar. Debe responderse y justificarse por escrito, revisando en particular el uso de cepas de microalgas."],
]
story.append(obligatorio_table(obligatorios_rows))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Adicionalmente, en la sección no sumativa de <b>Productos Académicos</b> (Cuadro 7 del formulario) "
    "no está marcada ninguna opción de producto (Opción 1, 2 o 3). Aunque esta sección no suma puntos a la "
    "calificación final, la rúbrica indica que si el equipo no cumple con declarar sus productos comprometidos, "
    "el Comité Técnico <b>debe</b> no recomendar la propuesta independientemente del puntaje obtenido.",
    styles["Alert"]
))

story.append(PageBreak())

# ---------- Secciones detalladas ----------
story.append(section_header("1. Grupo Investigador — peso 20%"))
story.append(bullets([
    "<b>Composición disciplinaria del equipo (5%):</b> la combinación de Física, Química y Biología es adecuada, pero no se documenta en el texto la experiencia previa de cada integrante en investigación sobre PAW/plasma (años, publicaciones). Sin esa evidencia no se puede alcanzar el nivel \"Excelente\".",
    "<b>Experiencia académica del equipo sobre el tema (4%):</b> no se listan productos académicos del equipo indexados en WoS/Scopus/SciELO relacionados con el tema. Agregar esa evidencia explícitamente en el cuerpo de la propuesta.",
    "<b>Articulación entre dependencias académicas (4%):</b> hay tres escuelas involucradas (Física, Química, Biología), lo cual es positivo, pero no se observan roles diferenciados por dependencia en el documento principal; verificar que sí queden explícitos en el plan de acción (Excel).",
    "<b>Vinculación externa (2%):</b> el único colaborador externo declarado tiene institución \"No aplica\" y no se describe su rol ni hay evidencia de compromiso formal. Formalizar su participación o ajustar la expectativa de puntaje en este criterio.",
    "<b>Trayectoria de la persona coordinadora (2%):</b> no se indican en el texto los años de experiencia, grado académico ni cantidad de proyectos coordinados por el Dr. Iván Vargas Blanco. Sin esta información explícita no se puede evaluar en el nivel más alto.",
    "<b>Participación estudiantil (3%, obligatorio):</b> ver tabla de requisitos obligatorios — actualmente ausente.",
]))

story.append(section_header("2. Fundamentación y Vacío de Conocimiento — peso 12%"))
story.append(bullets([
    "<b>Definición y justificación del problema (4%):</b> la Sección 4 conserva notas de trabajo sin desarrollar (\"Pequeña intro\", \"Nivel nacional\", \"Líneas de investigación\", \"objetivos estratégicos 2.2.3, 2.2.4, ODS 3,6,13,14,15\") en vez de párrafos redactados. Falta formular explícitamente las preguntas de investigación y desarrollar en prosa la vinculación con las líneas de investigación de la Escuela de Física, los objetivos estratégicos del ITCR y los ODS citados.",
    "<b>Coherencia entre problema y objetivos (4%):</b> los objetivos específicos están bien planteados y son medibles/verificables (CCD, LME, etc.). Este criterio está relativamente cerca del máximo; revisar la conexión explícita una vez se reescriba la Sección 4.",
    "<b>Estado del arte / reconocimiento de vacíos (4%, obligatorio):</b> los párrafos 5 (falta parte de un coautor), 7 (microalgas, con una nota pendiente \"{Introducir las variables que vayamos a medir…}\" aún dentro del texto) y 8 (lechugas) están marcados como parciales o con pendientes. El <b>párrafo 9 (\"Cierre e importancia del proyecto según el estado del arte\")</b>, que según el propio esquema del equipo es donde debía explicitarse el cierre, aparece únicamente como título, sin contenido redactado — es precisamente el espacio donde correspondería declarar el aporte original frente al antecedente más cercano [11]. Persisten además comentarios internos de revisión (Ui1, IV2–IV6, IV8, IV9, Ui7, Ui10) que señalan exactamente los mismos vacíos que exige la rúbrica. Deben resolverse y eliminarse antes de someter la propuesta.",
]))

story.append(section_header("3. Metodología — peso 15%"))
story.append(bullets([
    "<b>Rigor metodológico (8%, obligatorio):</b> OE1–OE3 están muy bien detallados, aunque en <b>OE3</b> la descripción del modelo de efectos mixtos queda cortada a media frase (\"la identificación de muestra y día se considerarán como efectos aleatorios para considerar la\" — el texto termina ahí, sin cerrar la idea). Falta indicar el modelo del medidor multiparamétrico de pH/EC/ORP (\"se utilizará un [poner nombre del modelo]\"), y hay una sección de \"Determinación de Nitrato\" duplicada que un comentario interno ya pidió eliminar pero sigue apareciendo dos veces. Existe también un error tipográfico en las fórmulas de rango operativo (V_min ≤ V ≤ V_min; el límite superior debería ser V_max, igual para f y D).",
    "<b>OE4 – microalgas:</b> el apartado \"4.1 Para evaluar el efecto del PAW en el crecimiento y desarrollo de microalgas\" dice literalmente \"AGREGAR\". Sin embargo, una revisión completa del documento muestra que la metodología de microalgas <b>sí existe</b> más adelante (numerada 1–6: cultivo de la microalga, preparación de la muestra, configuración del IRGA, curva de respuesta a la luz, curva de respuesta al CO2 y análisis de datos), pero quedó ubicada después de la sección de lechugas en lugar de bajo el encabezado 4.1, y el marcador \"AGREGAR\" nunca se reemplazó. Se recomienda reorganizar el documento moviendo ese contenido a su lugar correcto y eliminando el marcador, para que la propuesta se lea de forma coherente.",
    "<b>Metodología de lechugas (4.2.1):</b> el texto conserva varias decisiones metodológicas sin resolver, escritas como notas del propio equipo entre paréntesis: el tipo de tierra \"varía de a quién se la compren\", no se define si la siembra será individual o en \"camas\" (con la distancia entre plantas condicionada a esa decisión), la cantidad de riego \"se podría parametrizar, pero depende de varias cosas\" y queda pendiente decidir si el PAW se preparará con agua destilada o de tubo (con la advertencia de que el agua destilada podría bajar demasiado el pH y dañar las plantas). Estas decisiones deben cerrarse antes de someter la propuesta.",
    "<b>Definición de la población de estudio (7%, obligatorio):</b> no se indica el número total de corridas/repeticiones del CCD (OE2), ni cuántas muestras de PAW por reactor/condición se seguirán durante los 28 días de estabilidad (OE3, más allá de las 5 muestras control de agua destilada), ni el número de plantas de lechuga o réplicas por tratamiento/pozo de microalgas (OE4). Falta justificación estadística del tamaño muestral en los tres casos.",
]))

story.append(section_header("4. Plan de Acción y Cronograma — peso 13%"))
story.append(bullets([
    "Los tres criterios de esta sección son obligatorios y dependen enteramente del archivo Excel \"Plan de Acción - Gestión de Riesgos – 2027\", que no fue incluido en los documentos revisados, por lo que no se pudieron evaluar directamente.",
    "Verificar en ese archivo que cada actividad tenga responsable de ejecución, persona de apoyo (si aplica) y persona supervisora claramente diferenciadas.",
    "Verificar que las giras/trabajo de campo (por ejemplo, los traslados al CIB para medir nitritos y nitratos) estén planificadas con cronograma, justificación, producto esperado y presupuesto asociado.",
    "Verificar que cada actividad tenga un resultado esperado y un plazo definido, distinto de los productos académicos comprometidos, y que se incluya obligatoriamente la entrega de informes de avance (requisito explícito del formulario, Sección 8).",
]))

story.append(section_header("5. Presupuesto — peso 10%"))
story.append(bullets([
    "<b>Proporcionalidad y coherencia presupuestaria (4%, obligatorio):</b> el Cuadro 4 (monto total solicitado por año) está completamente vacío en este documento. El propio formulario indica además que el desglose detallado de la formulación presupuestaria se realiza en <b>un archivo de Excel adicional</b> (distinto del de Plan de Acción/Riesgos), que tampoco fue incluido en la revisión. No se puede verificar que el presupuesto respete los montos máximos de la convocatoria hasta contar con ambos documentos completos.",
    "<b>Inversión y equipamiento (4%, obligatorio):</b> solo aparece un enlace suelto a un barotermohigrómetro sin justificación formal ni monto, que parece una nota de trabajo olvidada en el texto. Formalizar la lista de equipo, su justificación y costo dentro del Excel de presupuesto, y confirmar que esté concentrada en el primer año de ejecución.",
    "<b>Financiamiento externo, contrapartida o cofinanciamiento (2%):</b> el Cuadro 5 también está vacío. Dado que en las preguntas 1.2 y 1.6 se indicó que no hay financiamiento externo ni convenio vigente, esto es esperable si en efecto no existe cofinanciamiento; aun así conviene dejarlo explícito (\"No aplica\") en vez de casillas en blanco.",
]))

story.append(section_header("6. Gestión de Riesgos — peso 8%"))
story.append(bullets([
    "<b>Identificación de riesgos por actividad (4%, obligatorio):</b> depende del archivo Excel no incluido en la revisión. Verificar que cada actividad del plan de acción tenga asociado al menos un riesgo con su respectiva acción de mitigación.",
    "<b>Viabilidad de las acciones de mitigación (4%, obligatorio):</b> es un requisito obligatorio distinto del anterior — no basta con que exista una acción de mitigación declarada; debe ser coherente, viable y suficiente para salvaguardar el éxito de cada actividad, con responsable de supervisión cuando aplique, no solo redacción genérica.",
]))

story.append(section_header("7. Aspectos Éticos — peso 7%"))
story.append(bullets([
    "<b>Evaluación de ente ético o regulatorio y justificación de exenciones (7%, obligatorio):</b> en la Sección 14 del formulario ninguna de las cuatro preguntas (a–d) está marcada, y la declaración de exención permanece sin redactar (\"Indique aquí las razones\").",
    "Dado que el proyecto trabaja con cepas de microalgas (Isochrysis galbana, Scenedesmus sp.), revisar con cuidado la pregunta (b) sobre uso de biodiversidad o material biológico silvestre: si las cepas provienen de una colección ya establecida probablemente no se requiera gestión ante CONAGEBIO, pero eso debe declararse y justificarse explícitamente por escrito, no dejarse en blanco.",
]))

story.append(section_header("8. Pertinencia e Impacto — peso 15%"))
story.append(bullets([
    "<b>Contribución disciplinaria e impacto (8%):</b> la Sección 4 ya conecta el proyecto con soberanía alimentaria y planes nacionales (MIDEPLAN, ODS). Para alcanzar el nivel \"Excelente\" (interés nacional <b>e internacional</b>) conviene reforzar explícitamente la relevancia internacional del tema, no solo la nacional.",
    "<b>Innovación del proyecto (7%):</b> mismo punto señalado en el estado del arte — falta articular con claridad qué aporta este proyecto que no esté ya cubierto por el antecedente de Patil y Chavan [11], y detallar las etapas de esa innovación. Es el hueco que el propio equipo identificó en sus comentarios internos.",
]))

story.append(section_header("Sección no sumativa: Productos Académicos"))
story.append(bullets([
    "El total de horas semanales (12) está bien calculado y marca correctamente la categoría \"11–20 horas → 2 opciones de producto\" en el Cuadro 6.",
    "Sin embargo, en el Cuadro 7 no está marcada ninguna opción de producto (Opción 1, 2 o 3). Sin esta selección el Comité Técnico no puede verificar el tipo de producto comprometido, lo que según la rúbrica puede derivar en que la propuesta no sea recomendada, independientemente del puntaje obtenido en las demás secciones.",
]))

story.append(section_header("Otros pendientes de forma (no puntúan directo en la rúbrica, pero deben resolverse)"))
story.append(bullets([
    "Resumen (Sección 2) y Abstract/Keywords en inglés están sin redactar; solo las palabras clave en español están completas.",
    "Sección 12 (\"Resguardo y depósito de productos\"): ninguna de las dos casillas de declaración (a y b) está marcada.",
    "Sección 13 (estrategia de divulgación y socialización) y Sección 15 (hoja de ruta) están sin completar.",
    "Quedan comentarios de Word visibles en el documento (IV1–IV9, Ui1, Ui7, Ui10) que deben resolverse y eliminarse antes de enviar la propuesta.",
    "Firma y número de cédula de la persona coordinadora (Sección 17, declaración jurada) están sin completar.",
    "Sección 11c (\"Otros resultados y activos tecnológicos, complementarios\"): los dos campos de producto adicional conservan el texto de ejemplo del formulario (\"ej. Software, prototipo...\", \"ej. Guía técnica, manual...\") sin que el equipo los haya completado o marcado como no aplicable.",
    "Desfase menor de fechas: el proyecto declara finalización el 21/12/2028, pero el período de nombramiento del equipo en el Cuadro 2 llega hasta el 31/12/2028 (10 días de diferencia). Revisar cuál fecha es la correcta.",
    "Posible inconsistencia menor a revisar: la pregunta 1.9 declara que la propuesta NO presenta potencial de propiedad intelectual, mientras que la 1.10 declara que SÍ se espera una innovación (de proceso). No son necesariamente contradictorias, pero conviene revisar que ambas respuestas sean coherentes entre sí antes de enviar.",
]))

story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#999999"), spaceBefore=4, spaceAfter=6))
story.append(Paragraph(
    "Documento generado a partir del cruce entre la Rúbrica de Evaluación – Investigación Básica y Aplicada "
    "(Convocatoria VIE 2027) y el formulario de propuesta \"Desarrollo y evaluación comparativa de reactores "
    "corona y DBD para la producción de agua activada por plasma con aplicaciones en el cultivo de microalgas "
    "y lechuga\".",
    styles["TocNote"]
))

doc = SimpleDocTemplate(
    "/home/claude/informe/Observaciones_Propuesta_VIE2027_Lab_Plasmas.pdf",
    pagesize=letter,
    leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
    title="Observaciones - Propuesta VIE 2027 - Laboratorio de Plasmas"
)
doc.build(story)
print("PDF generado correctamente")