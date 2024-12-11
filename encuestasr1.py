import pandas as pd
import streamlit as st

# Leer el archivo Excel
df = pd.read_excel('preguntas.xlsx')

# Eliminar espacios extra en los nombres de las columnas
df.columns = df.columns.str.strip()

# Mostrar las primeras filas para verificar la lectura
st.write("Preguntas cargadas:")
st.write(df.head())  # Muestra las primeras filas para confirmar

# Mostrar el formulario dinámicamente
for index, row in df.iterrows():
    if row['escala'] == 'likert':
        # Separar las opciones de la escala y asignar a un diccionario
        opciones = row['posibles respuestas'].split(', ')
        labels = [opcion.split(': ')[1] for opcion in opciones]
        valores = [opcion.split(': ')[0] for opcion in opciones]

        # Mostrar la pregunta con la escala Likert
        seleccion = st.radio(row['pregunta'], labels, index=0)

        # Mostrar la respuesta seleccionada
        respuesta = valores[labels.index(seleccion)]
        st.write(f"Respuesta seleccionada: {respuesta}")

    elif row['escala'] == 'binario':
        # Separar las opciones binarias
        opciones = row['posibles respuestas'].split(', ')
        labels = [opcion.split(': ')[1] for opcion in opciones]
        valores = [opcion.split(': ')[0] for opcion in opciones]

        # Mostrar la pregunta con las opciones binarias
        seleccion = st.radio(row['pregunta'], labels, index=0)

        # Mostrar la respuesta seleccionada
        respuesta = valores[labels.index(seleccion)]
        st.write(f"Respuesta seleccionada: {respuesta}")

    elif row['escala'] == 'triple':
        # Separar las opciones de tres respuestas
        opciones = row['posibles respuestas'].split(', ')
        labels = [opcion.split(': ')[1] for opcion in opciones]
        valores = [opcion.split(': ')[0] for opcion in opciones]

        # Mostrar la pregunta con tres opciones
        seleccion = st.radio(row['pregunta'], labels, index=0)

        # Mostrar la respuesta seleccionada
        respuesta = valores[labels.index(seleccion)]
        st.write(f"Respuesta seleccionada: {respuesta}")
