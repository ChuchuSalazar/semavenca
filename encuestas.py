import streamlit as st
import pandas as pd

# Cargar las preguntas desde el archivo de Excel sin encabezados
df_preguntas = pd.read_excel('preguntas.xlsx', header=None)

# Asignar nombres a las columnas manualmente, ya que no hay encabezados
df_preguntas.columns = ['item', 'pregunta', 'escala', 'posibles_respuestas']

# Función para mostrar la encuesta


def mostrar_encuesta():
    respuestas = {}
    preguntas_no_respondidas = []

    for i, row in df_preguntas.iterrows():
        # Acceder a las columnas por su nombre después de asignar los nombres manualmente
        pregunta_id = row['item']  # Columna A (ITEM)
        pregunta_texto = row['pregunta']  # Columna B (PREGUNTA)
        escala = row['escala']  # Columna C (ESCALA)
        posibles_respuestas = row['posibles_respuestas'].split(
            ",")  # Columna D (POSIBLES_RESPUESTAS)

        # Mostrar la pregunta en Streamlit
        st.write(f"{pregunta_id}. {pregunta_texto}")

        # Crear las opciones basadas en la escala
        opciones = []
        for opcion in posibles_respuestas:
            num, texto = opcion.split(":")
            opciones.append(f"{num.strip()}: {texto.strip()}")

        # Usar selectbox o radio button según la escala
        if int(escala) == 2:
            respuesta = st.radio(
                f"Seleccione: {pregunta_texto}", options=["Sí", "No"])
        elif int(escala) == 3:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "De acuerdo", "Neutral", "En desacuerdo"])
        elif int(escala) == 4:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "Muy en desacuerdo", "En desacuerdo", "De acuerdo", "Muy de acuerdo"])
        elif int(escala) == 5:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"])

        respuestas[pregunta_id] = respuesta

    # Validar que se hayan respondido todas las preguntas
    for pregunta_id, respuesta in respuestas.items():
        if respuesta is None:
            preguntas_no_respondidas.append(pregunta_id)

    if preguntas_no_respondidas:
        st.warning(f"Las siguientes preguntas no han sido respondidas: {
                   ', '.join(preguntas_no_respondidas)}")
    else:
        st.success("Gracias por completar la encuesta.")


# Llamar a la función para mostrar la encuesta
mostrar_encuesta()
