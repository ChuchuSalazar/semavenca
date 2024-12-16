import streamlit as st
import pandas as pd
import random
import datetime
from firebase_admin import credentials, firestore, initialize_app, get_app
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

# Inicializar Firebase solo una vez
try:
    app = get_app()
except ValueError:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    initialize_app(cred)

# Conectar a Firestore
db = firestore.client()

# Generar un ID único


def generar_id():
    return random.randint(100000, 999999)


# URL del archivo de preguntas
url_preguntas = 'https://raw.githubusercontent.com/ChuchuSalazar/encuesta/main/preguntas.xlsx'

# Cargar preguntas
df_preguntas = pd.read_excel(url_preguntas, header=None)
df_preguntas.columns = ['item', 'pregunta', 'escala', 'posibles_respuestas']

# Guardar respuestas en Firebase


def guardar_respuestas(respuestas):
    id_encuesta = f"ID_{generar_id()}"
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {'FECHA': fecha}
    for key, value in respuestas.items():
        data[key] = value

    db.collection('respuestas').document(id_encuesta).set(data)

# Mostrar encuesta


def mostrar_encuesta():
    st.title("Encuesta de Hábitos de Ahorro")
    st.write("Por favor, responda todas las preguntas obligatorias.")

    # Diccionario para respuestas
    respuestas = {}
    preguntas_faltantes = []  # Para rastrear preguntas sin responder

    # Sección de preguntas
    st.header("Preguntas de la Encuesta")
    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = int(row['escala'])
        opciones = row['posibles_respuestas'].split(',')[:escala]

        # Validación dinámica: marcar preguntas no respondidas
        estilo_borde = "2px solid blue"  # Borde azul por defecto
        if st.session_state.get(f"respuesta_{pregunta_id}", None) is None and pregunta_id in preguntas_faltantes:
            estilo_borde = "3px solid red"  # Borde rojo si falta responder

        # Mostrar la pregunta con estilo de borde
        st.markdown(
            f"""<div style="border: {estilo_borde}; padding: 10px; border-radius: 5px;">
                    {pregunta_texto}
                </div>""",
            unsafe_allow_html=True,
        )

        # Crear opciones de respuesta
        respuesta = st.radio(
            f"Seleccione una opción para la Pregunta {i+1}:",
            opciones,
            index=None,  # No hay selección por defecto
            key=f"respuesta_{pregunta_id}",
        )
        respuestas[pregunta_id] = respuesta

    # Botón para enviar
    if st.button("Enviar"):
        preguntas_faltantes.clear()

        # Validar respuestas
        for i, row in df_preguntas.iterrows():
            pregunta_id = row['item']
            if respuestas[pregunta_id] is None:
                preguntas_faltantes.append((i + 1, pregunta_id))

        # Si hay preguntas faltantes, mostrar un modal con un mensaje
        if preguntas_faltantes:
            faltantes = ", ".join([str(num_pregunta)
                                  for num_pregunta, _ in preguntas_faltantes])
            st.error(
                "❗ Por favor, responda las preguntas resaltadas en rojo antes de continuar.")

            # Mostrar una ventana modal personalizada
            st.markdown(
                f"""
                <div style="position: fixed; top: 20%; left: 30%;
                background-color: #f8d7da; color: #721c24; padding: 20px;
                #f5c6cb; box-shadow: 2px 2px 10px gray;">
                border-radius: 10px; border: 2px solid
                    <h4 style="margin-bottom: 10px;">Preguntas Sin Responder</h4>
                    <p>Faltan por responder las siguientes preguntas: <b>{faltantes}</b></p>
                    <button onclick="window.location.reload()"
                    style="background-color: #007bff; color: white; border: none;
                    padding: 10px; border-radius: 5px; cursor: pointer;">
                    Aceptar</button>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # Guardar las respuestas en Firebase
            guardar_respuestas(respuestas)
            st.success("¡Gracias por completar la encuesta!")
            st.balloons()

            # Bloquear preguntas después del envío
            st.write("La encuesta ha sido enviada exitosamente.")
            st.stop()


# Ejecutar la encuesta
if __name__ == '__main__':
    mostrar_encuesta()
