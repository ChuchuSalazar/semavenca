import random
import datetime
import os
import streamlit as st
import pandas as pd
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Cargar las preguntas desde el archivo de Excel
# Asegúrate de que la primera fila son encabezados
df_preguntas = pd.read_excel('preguntas.xlsx', header=0)

# Inicializar Firebase
# Ruta a las credenciales de Firebase
firebase_creds = os.getenv('FIREBASE_CREDS_PATH')
cred = credentials.Certificate(firebase_creds)
initialize_app(cred)
db = firestore.client()

# Función para mostrar la encuesta


def mostrar_encuesta():
    respuestas = {}
    preguntas_no_respondidas = []

    # Aleatorizar el orden de las preguntas
    df_preguntas = df_preguntas.sample(frac=1).reset_index(drop=True)

    # Mostrar "Información General"
    st.subheader("Información General")
    st.write("Por favor, complete la encuesta a continuación.")

    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = int(row['escala'])  # Escala debe ser convertido a int
        posibles_respuestas = row['posibles_respuestas'].split(
            ",")  # Opciones de respuesta

        if i == 5:
            st.subheader("Responda")

        st.markdown(f"**Pregunta {i + 1}:** {pregunta_texto}")

        # Dependiendo de la escala, se presentan diferentes tipos de respuestas
        if escala == 2:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "Sí", "No"], key=f"q{i}", disabled=False)
        elif escala == 3:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "De acuerdo", "Neutral", "En desacuerdo"], key=f"q{i}", disabled=False)
        elif escala == 4:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "Muy en desacuerdo", "En desacuerdo", "De acuerdo", "Muy de acuerdo"], key=f"q{i}", disabled=False)
        elif escala == 5:
            respuesta = st.radio(f"Seleccione: {pregunta_texto}", options=[
                                 "Totalmente en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Totalmente de acuerdo"], key=f"q{i}", disabled=False)

        respuestas[pregunta_id] = respuesta

        if respuesta is None:
            preguntas_no_respondidas.append(pregunta_id)

    # Si todas las preguntas fueron respondidas, guardar las respuestas
    if not preguntas_no_respondidas:
        st.success("Gracias por completar la encuesta.")
        st.balloons()

        # Registrar la fecha y hora de la encuesta
        fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Guardar las respuestas en Firebase
        doc_ref = db.collection('respuestas').document()
        doc_ref.set({
            'respuestas': respuestas,
            'fecha': fecha_actual
        })

    else:
        # Si no se han respondido todas las preguntas
        st.warning(
            "Por favor, responde todas las preguntas. Las preguntas no respondidas están marcadas en rojo.")


# Llamar a la función para mostrar la encuesta
mostrar_encuesta()
