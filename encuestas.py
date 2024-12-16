import pandas as pd
import datetime
from firebase_admin import firestore
import random

# Función para guardar las respuestas en Firebase


def guardar_respuestas(respuestas):
    # Generar un ID único para la encuesta
    id_encuesta = generar_id()
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Crear un diccionario con los datos adicionales
    data = {
        'ID': id_encuesta,
        'FECHA': fecha,
        'SEXO': respuestas.get('sexo', ''),
        'RANGO_EDA': respuestas.get('rango_edad', ''),
        'RANGO_INGRESO': respuestas.get('rango_ingreso', ''),
        'CIUDAD': respuestas.get('ciudad', ''),
        'NIVEL_PROF': respuestas.get('nivel_educativo', ''),
    }

    # Leer el archivo de preguntas con la información de las escalas
    df_preguntas = pd.read_excel('preguntas.xlsx')

    # Orden estricto de las preguntas y sesgos
    preguntas_ordenadas = [
        "AV1", "AV2", "AV3", "AV4", "AV5",  # Aversión a la pérdida
        "SQ1", "SQ2", "SQ3", "SQ4", "SQ5",  # Status quo
        "DH1", "DH2", "DH3", "DH4", "DH5",  # Descuento hiperbólico
        "CM1", "CM2", "CM3", "CM4", "CM5"   # Contabilidad mental
    ]

    # Mapear las respuestas de las escalas a sus valores numéricos
    def escala_a_numero(escala, respuesta_texto):
        opciones = escala.split(",")
        try:
            # Verificar si la respuesta está dentro de las opciones disponibles
            if respuesta_texto not in opciones:
                raise ValueError(f"Respuesta no válida: {
                                 respuesta_texto} para la escala {escala}")
            return opciones.index(respuesta_texto) + 1  # Convertir a número
        except ValueError:
            return None

    # Añadir las respuestas de las preguntas al diccionario
    for pregunta in preguntas_ordenadas:
        # Obtener la respuesta en texto
        respuesta_texto = respuestas.get(pregunta, '')

        # Buscar el tipo de escala y las opciones para la pregunta en el DataFrame
        pregunta_info = df_preguntas[df_preguntas['item'] == pregunta]
        if not pregunta_info.empty:
            # Obtener las opciones de respuesta
            escala = pregunta_info.iloc[0]['posibles_respuestas']
        else:
            escala = 'No disponible'

        # Convertir la respuesta según el tipo de escala
        respuesta_numerica = escala_a_numero(escala, respuesta_texto)

        if respuesta_numerica is None:
            # Si alguna pregunta no fue respondida o tiene una respuesta inválida
            raise ValueError(
                f"La pregunta {pregunta} no tiene una respuesta válida.")
        data[pregunta] = respuesta_numerica

    # Guardar los datos estrictos en Firebase
    db.collection('respuestas').document(str(id_encuesta)).set(data)

    print(f"Respuestas de la encuesta {id_encuesta} guardadas correctamente.")

# Función para generar un ID único


def generar_id():
    return random.randint(100000, 999999)

# Ejemplo de uso en tu flujo de preguntas


def mostrar_encuesta():
    respuestas = {}

    # Cargar las preguntas desde el archivo de Excel
    df_preguntas = pd.read_excel('preguntas.xlsx')

    # Mostrar las preguntas y respuestas
    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = row['posibles_respuestas'].split(',')

        # Mostrar la pregunta
        respuesta = st.radio(
            f"**Pregunta {i+1}:** {pregunta_texto}", escala, key=f'AV{pregunta_id}')
        respuestas[f'AV{pregunta_id}'] = respuesta

    # Botón para enviar las respuestas
    if st.button("Enviar"):
        guardar_respuestas(respuestas)
        st.balloons()
        st.success(
            "Gracias por completar la encuesta. ¡Tu respuesta ha sido registrada!")
