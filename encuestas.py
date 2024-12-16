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

# Inicializar Firebase solo si no está inicializado
try:
    app = get_app()
except ValueError:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    app = initialize_app(cred)

db = firestore.client(app)

# Función para generar un ID aleatorio


def generar_id():
    return random.randint(100000, 999999)


# Cargar las preguntas del archivo de Excel
df_preguntas = pd.read_excel('preguntas.xlsx')

# Función para guardar las respuestas en Firebase


def guardar_respuestas(respuestas):
    id_encuesta = generar_id()
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Crear un diccionario con todas las respuestas y los datos adicionales
    data = {
        'ID': id_encuesta,
        'FECHA': fecha,
        'SEXO': respuestas.get('sexo', ''),
        'RANGO_EDA': respuestas.get('rango_edad', ''),
        'RANGO_INGRESO': respuestas.get('rango_ingreso', ''),
        'CIUDAD': respuestas.get('ciudad', ''),
        'NIVEL_PROF': respuestas.get('nivel_educativo', ''),
    }

    # Añadir las respuestas de las preguntas
    for i, pregunta_id in enumerate(df_preguntas['item']):
        # Guardar como número
        data[f'AV{i+1}'] = int(respuestas.get(f'AV{pregunta_id}', 0))

    # Guardar en Firebase
    db.collection('respuestas').document(str(id_encuesta)).set(data)

# Función para mostrar la encuesta


def mostrar_encuesta():
    respuestas = {}

    # Mostrar el logo de la universidad
    st.image('logo_ucab.jpg', width=150,
             caption="Universidad Católica Andrés Bello")

    # Mostrar los datos demográficos
    st.header("Datos Demográficos")

    sexo = st.radio("Sexo:", ['1 - Masculino', '2 - Femenino',
                    '3 - Otro'], key='sexo', horizontal=True)
    respuestas['sexo'] = sexo.split()[0]

    rango_edad = st.radio("Rango de edad:", [
        '1 - 18-25', '2 - 26-35', '3 - 36-45', '4 - 46-60', '5 - Más de 60'], key='rango_edad', horizontal=True)
    respuestas['rango_edad'] = rango_edad.split()[0]

    rango_ingreso = st.radio("Rango de ingresos (US$):", [
        '1 - 0-300', '2 - 301-700', '3 - 701-1100', '4 - 1101-1500', '5 - 1501-3000', '6 - Más de 3000'], key='rango_ingreso', horizontal=True)
    respuestas['rango_ingreso'] = rango_ingreso.split()[0]

    ciudad = st.selectbox("Ciudad:", ['1 - Ciudad A', '2 - Ciudad B',
                          '3 - Ciudad C', '4 - Ciudad D', '5 - Ciudad E'], key='ciudad')
    respuestas['ciudad'] = ciudad.split()[0]

    nivel_educativo = st.radio("Nivel educativo:", [
        '1 - Primaria', '2 - Secundaria', '3 - Universitario', '4 - Postgrado'], key='nivel_educativo', horizontal=True)
    respuestas['nivel_educativo'] = nivel_educativo.split()[0]

    # Mostrar las preguntas numeradas y enmarcadas
    st.header("Preguntas de la Encuesta")

    for i, row in df_preguntas.iterrows():
        pregunta_id = row['item']
        pregunta_texto = row['pregunta']
        escala = ['1: Totalmente en desacuerdo', '2: En desacuerdo',
                  '3: Neutral', '4: De acuerdo', '5: Totalmente de acuerdo']

        st.markdown(f"**Pregunta {i+1}:**")
        st.markdown(f'<div style="border: 2px solid #add8e6; padding: 10px; border-radius: 5px; font-size: 16px; font-family: Arial, sans-serif;">{
                    pregunta_texto}</div>', unsafe_allow_html=True)

        respuesta = st.radio(f"", escala, key=f'AV{pregunta_id}')
        if respuesta != 'No seleccionar':
            respuestas[f'AV{pregunta_id}'] = respuesta.split(
                ':')[0]  # Extraer solo el número

    # Botón para enviar las respuestas
    if st.button("Enviar"):
        preguntas_faltantes = [f"Pregunta {
            i+1}" for i, row in df_preguntas.iterrows() if not respuestas.get(f'AV{row["item"]}', None)]

        if preguntas_faltantes:
            st.error(f"Por favor, responde las siguientes preguntas: {
                     ', '.join(preguntas_faltantes)}")
        else:
            guardar_respuestas(respuestas)
            st.balloons()
            st.success(
                "Gracias por completar la encuesta. ¡Tu respuesta ha sido registrada!")


# Llamar la función para mostrar la encuesta
if __name__ == '__main__':
    mostrar_encuesta()
