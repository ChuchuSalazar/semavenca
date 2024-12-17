import streamlit as st
import pandas as pd
import random
import datetime
from firebase_admin import credentials, initialize_app, db, get_app
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
    initialize_app(cred, {
        # Asegúrate de poner tu URL de Firebase Realtime Database
        "databaseURL": "https://tu-proyecto.firebaseio.com"
    })

# Conectar a Realtime Database
ref = db.reference("/respuestas")

# Generar un ID único


def generar_id():
    return random.randint(100000, 999999)


# URL del archivo de preguntas
url_preguntas = 'https://raw.githubusercontent.com/ChuchuSalazar/encuesta/main/preguntas.xlsx'

# Cargar preguntas
df_preguntas = pd.read_excel(url_preguntas, header=None)
df_preguntas.columns = ['item', 'pregunta', 'escala', 'posibles_respuestas']

# Guardar respuestas en Realtime Database


def guardar_respuestas(respuestas):
    id_encuesta = f"ID_{generar_id()}"
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    data = {'FECHA': fecha}
    for key, value in respuestas.items():
        data[key] = value

    # Subir datos a Realtime Database
    ref.child(id_encuesta).set(data)

# Mostrar encuesta


def mostrar_encuesta():
    st.title("Encuesta de Hábitos de Ahorro")
    st.write("Por favor, complete todas las preguntas obligatorias.")

    # Diccionario para respuestas
    respuestas = {}
    preguntas_faltantes = []

    # Sección demográfica
    st.header("Información Demográfica")
    datos_demograficos = {
        'Sexo': ['Masculino', 'Femenino', 'Otro'],
        'Ciudad': ['Ciudad A', 'Ciudad B', 'Ciudad C'],
        'Escala de Edad': ['18-25', '26-35', '36-50', '51+'],
        'Escala de Ingresos': ['Bajo', 'Medio', 'Alto'],
        'Nivel Profesional': ['Primaria', 'Secundaria', 'Universitario', 'Postgrado']
    }

    for pregunta, opciones in datos_demograficos.items():
        if pregunta == "Ciudad":
            respuesta = st.selectbox(
                f"**{pregunta}**",
                ['Seleccione una opción'] + opciones,
                index=0,
                key=f"demografico_{pregunta}"
            )
            if respuesta == 'Seleccione una opción':
                respuestas[pregunta] = None
            else:
                respuestas[pregunta] = respuesta
        else:
            respuesta = st.multiselect(
                f"**{pregunta}**", opciones, key=f"demografico_{pregunta}")
            if len(respuesta) == 0:
                respuestas[pregunta] = None
            else:
                respuestas[pregunta] = ', '.join(respuesta)

    # Preguntas principales
    st.header("Preguntas de la Encuesta")
    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = int(row['escala'])
        opciones = row['posibles_respuestas'].split(',')[:escala]

        # Estilo dinámico del borde (azul o rojo si faltante)
        estilo_borde = "2px solid blue"
        if st.session_state.get(f"respuesta_{pregunta_id}_faltante", False):
            estilo_borde = "3px solid red"

        # Pregunta
        st.markdown(
            f"""
            <div style="border: {estilo_borde}; padding: 10px; margin-bottom: 10px;
                        border-radius: 5px; background-color: #f9f9f9;">
                {pregunta_texto}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Opciones
        respuesta = st.radio(
            f"Pregunta {i+1}:",
            opciones,
            index=None,
            key=f"respuesta_{pregunta_id}"
        )
        respuestas[pregunta_id] = respuesta

    # Botón de enviar
    if st.button("Enviar"):
        preguntas_faltantes.clear()

        # Validar preguntas no respondidas
        for key, value in respuestas.items():
            if value is None:
                preguntas_faltantes.append(key)
                st.session_state[f"{key}_faltante"] = True
            else:
                st.session_state[f"{key}_faltante"] = False

        # Ventana emergente si hay preguntas faltantes
        if preguntas_faltantes:
            st.markdown(
                f"""
                <script>
                    alert(
                        "Por favor, responda las siguientes preguntas: {', '.join([str(f) for f in preguntas_faltantes])}");
                </script>
                """,
                unsafe_allow_html=True
            )
        else:
            guardar_respuestas(respuestas)
            st.success("¡Gracias por completar la encuesta!")
            st.balloons()
            st.stop()


# Ejecutar
if __name__ == '__main__':
    mostrar_encuesta()
