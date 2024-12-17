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

# Inicializar Firebase solo si no se ha inicializado previamente
try:
    app = get_app()
except ValueError as e:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    app = initialize_app(cred)

# Conectar a Firestore
db = firestore.client()

# Función para generar un ID único


def generar_id():
    return random.randint(100000, 999999)


# URL del archivo de preguntas
url_preguntas = 'https://raw.githubusercontent.com/ChuchuSalazar/encuesta/main/preguntas.xlsx'

# Función para cargar preguntas


def cargar_preguntas(url):
    try:
        # Leer encabezados desde la primera fila
        df = pd.read_excel(url, header=0)
        columnas_esperadas = ['item', 'pregunta',
                              'escala', 'posibles_respuestas']
        if not all(col in df.columns for col in columnas_esperadas):
            st.error(
                "El archivo no contiene las columnas esperadas: 'item', 'pregunta', 'escala', 'posibles_respuestas'")
            st.stop()

        df['escala'] = pd.to_numeric(df['escala'], errors='coerce')
        df = df.dropna(subset=['escala'])
        df['escala'] = df['escala'].astype(int)
        return df
    except Exception as e:
        st.error(f"Error al cargar las preguntas: {e}")
        st.stop()


# Cargar preguntas desde el archivo
df_preguntas = cargar_preguntas(url_preguntas)

# Función para guardar respuestas en Firebase


def guardar_respuestas(respuestas):
    id_encuesta = f"ID_{generar_id()}"
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {'FECHA': fecha}
    data.update(respuestas)
    db.collection('respuestas').document(id_encuesta).set(data)

# Función principal para mostrar la encuesta


def mostrar_encuesta():
    st.title("Encuesta de Hábitos de Ahorro")
    st.write("Por favor, responda todas las preguntas obligatorias.")

    # Diccionario para respuestas
    respuestas = {}

    # Preguntas iniciales: Datos demográficos
    st.header("Datos Demográficos")

    # Distribuir Sexo y Rango de Edad horizontalmente
    col1, col2 = st.columns(2)

    with col1:
        respuestas['sexo'] = st.radio(
            "Sexo", ["Masculino", "Femenino", "Otro"], index=None, horizontal=True)

    with col2:
        respuestas['rango_edad'] = st.radio(
            "Rango de Edad", ["18-25", "26-35", "36-45", "46-60", "60+"], index=None, horizontal=True
        )

    # Pregunta de Rango de Ingresos
    respuestas['rango_ingresos'] = st.radio(
        "Rango de Ingresos Mensuales",
        ["Menos de $500", "$500-$1000", "$1000-$2000", "Más de $2000"],
        index=None
    )

    # Nivel de Educación y Ciudad (desplegables)
    respuestas['nivel_educacion'] = st.selectbox(
        "Nivel de Educación",
        ["Seleccione una opción", "Primaria", "Secundaria",
            "Pregrado", "Posgrado", "Doctorado"],
        index=0
    )

    respuestas['ciudad'] = st.text_input(
        "Ciudad de Residencia", placeholder="Ingrese su ciudad aquí")

    # Validación de preguntas iniciales
    if st.button("Continuar a la Encuesta"):
        if None in respuestas.values() or respuestas['nivel_educacion'] == "Seleccione una opción" or respuestas['ciudad'].strip() == "":
            st.error(
                "Por favor, complete todos los datos demográficos antes de continuar.")
            st.stop()
        else:
            st.success("¡Datos demográficos completados!")
            # Marcar como completado
            st.session_state['datos_completados'] = True

    # Continuar solo si los datos demográficos están completos
    if 'datos_completados' in st.session_state and st.session_state['datos_completados']:
        st.header("Preguntas de la Encuesta")
        preguntas_faltantes = []  # Rastreo de preguntas sin respuesta

        # Mostrar preguntas del Excel
        for i, row in df_preguntas.iterrows():
            pregunta_id = row['item']
            pregunta_texto = row['pregunta']
            escala = int(row['escala'])
            opciones = row['posibles_respuestas'].split(',')[:escala]

            respuesta = st.radio(
                f"{pregunta_texto}",
                opciones,
                index=None,
                key=f"respuesta_{pregunta_id}"
            )
            respuestas[pregunta_id] = respuesta

        # Botón para enviar respuestas
        if st.button("Enviar Encuesta"):
            preguntas_faltantes = [
                k for k, v in respuestas.items() if v is None]
            if preguntas_faltantes:
                st.error(
                    "Por favor, responda todas las preguntas antes de enviar la encuesta.")
            else:
                guardar_respuestas(respuestas)
                st.success("¡Gracias por completar la encuesta!")
                st.balloons()
                st.write("La encuesta ha sido enviada exitosamente.")
                st.stop()


# Ejecutar la encuesta
if __name__ == '__main__':
    mostrar_encuesta()
