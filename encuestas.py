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

    # Preguntas principales
    st.header("Preguntas de la Encuesta")
    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = int(row['escala'])
        opciones = row['posibles_respuestas'].split(',')[:escala]

        # Estilo dinámico de borde (inicialmente azul)
        estilo_borde = "2px solid blue"
        if st.session_state.get(f"respuesta_{pregunta_id}", None) is None:
            estilo_borde = "2px solid blue"

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

        # Cambiar estilo si hay faltantes
        for i, row in df_preguntas.iterrows():
            pregunta_id = row['item']
            if pregunta_id in preguntas_faltantes:
                st.session_state[f"respuesta_{
                    pregunta_id}_borde"] = "3px solid red"

        # Ventana emergente simulada si hay preguntas faltantes
        if preguntas_faltantes:
            faltantes_texto = ", ".join([str(f) for f in preguntas_faltantes])
            st.markdown(
                f"""
                <div style="
                    position: fixed; top: 30%; left: 20%; right: 20%; z-index: 9999;
                    background-color: white; padding: 20px; text-align: center;
                    border: 3px solid red; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
                    border-radius: 10px;">
                    <h4 style="color: red;">❗ Preguntas Sin Responder</h4>
                    <p>Por favor, responda las siguientes preguntas: <b>{faltantes_texto}</b></p>
                    <button onclick="window.location.reload()"
                        style="background-color: #007bff; color: white; padding: 10px 20px;
                        border: none; border-radius: 5px; cursor: pointer;">
                        Aceptar
                    </button>
                </div>
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
