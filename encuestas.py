import streamlit as st
import pandas as pd
import random
from datetime import datetime
import time

# Cargar el archivo de preguntas
df = pd.read_excel("preguntas.xlsx")

# Función para generar un ID aleatorio para cada encuesta


def generar_id():
    return str(random.randint(100000, 999999))

# Mostrar la fecha y hora actual


def mostrar_fecha_hora():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Función para mostrar los datos demográficos


def mostrar_datos_demograficos():
    st.markdown("""
        <div style="background-color: lightgray; padding: 15px; border-radius: 5px;">
            <h3>Por favor, complete los datos demográficos.</h3>
        </div>
    """, unsafe_allow_html=True)

    # Caja para el ID de control
    id_encuesta = generar_id()
    st.text_input("Número de Control", id_encuesta, disabled=True)

    # Sexo
    sexo = st.radio("Sexo", ["Hombre", "Mujer"], key="sexo")

    # Rango de edad
    edad = st.selectbox(
        "Rango de Edad", ["18-25", "26-35", "36-45", "46-60", "60+"], key="edad")

    # Rango de salario
    salario_min = st.number_input(
        "Salario Mínimo", min_value=0, step=100, key="salario_min")
    salario_max = st.number_input(
        "Salario Máximo", min_value=0, step=100, key="salario_max")

    # Nivel Educativo
    nivel_educativo = st.selectbox("Nivel Educativo", [
                                   "Primaria", "Secundaria", "Universidad", "Postgrado"], key="educativo")

    # Ciudad
    ciudad = st.selectbox(
        "Ciudad", ["Caracas", "Maracaibo", "Valencia", "Barquisimeto"], key="ciudad")

    return {
        "id_encuesta": id_encuesta,
        "sexo": sexo,
        "edad": edad,
        "salario": (salario_min, salario_max),
        "nivel_educativo": nivel_educativo,
        "ciudad": ciudad
    }

# Función para mostrar las preguntas


def mostrar_preguntas():
    preguntas_respondidas = []
    preguntas_no_respondidas = []
    respuestas = {}

    for index, row in df.iterrows():
        # Mostrar la pregunta
        pregunta_estilo = 'background-color: lightblue; padding: 10px; border: 2px solid blue; border-radius: 5px; margin-bottom: 10px;'
        pregunta_no_respondida_estilo = 'background-color: lightblue; padding: 10px; border: 2px solid red; border-radius: 5px; margin-bottom: 10px;'

        # Mostrar la pregunta
        st.markdown(f'<div style="{pregunta_estilo}"><strong>{
                    index + 1}. {row["pregunta"]}</strong></div>', unsafe_allow_html=True)

        # Mostrar las opciones
        opciones = row['posibles_respuestas'].split(',')
        seleccionada = st.radio(f"Respuesta a la pregunta {
                                index + 1}", opciones, key=f"q{index}")

        # Guardar la respuesta
        respuestas[index] = seleccionada

        # Comprobar si la respuesta está seleccionada
        if seleccionada:
            preguntas_respondidas.append(index)
        else:
            preguntas_no_respondidas.append(index)

    return respuestas, preguntas_respondidas, preguntas_no_respondidas

# Función para mostrar el resumen


def mostrar_resumen():
    respuestas, preguntas_respondidas, preguntas_no_respondidas = mostrar_preguntas()

    # Validar que todas las preguntas estén respondidas
    if len(preguntas_no_respondidas) > 0:
        st.markdown('<div style="background-color: #f2f2f2; padding: 15px; border-radius: 5px;"><strong style="color:red;">Por favor, responda todas las preguntas.</strong></div>', unsafe_allow_html=True)

        # Mostrar las preguntas no respondidas en rojo
        for index in preguntas_no_respondidas:
            st.markdown(f'<div style="background-color: lightblue; padding: 10px; border: 2px solid red; border-radius: 5px; margin-bottom: 10px;">{
                        index + 1}. {df.iloc[index]["pregunta"]}</div>', unsafe_allow_html=True)

    else:
        # Si todas las preguntas fueron respondidas, mostrar mensaje de agradecimiento
        st.markdown('<div style="background-color: lightgreen; padding: 15px; border-radius: 5px;"><strong>Gracias por participar en la encuesta. Todos los campos han sido respondidos.</strong></div>', unsafe_allow_html=True)

    return respuestas

# Función principal


def app():
    st.title("Encuesta de Investigación")

    # Mostrar el logo de UCAB alineado a la derecha
    st.image("ucab_logo.jpg", width=150, use_column_width=False,
             caption="UCAB", use_container_width=False)

    # Mostrar la fecha y hora de inicio
    st.write(f"Fecha y hora de llenado: {mostrar_fecha_hora()}")

    # Mostrar las instrucciones
    st.markdown("""
        <div style="background-color: lightgray; padding: 15px; border-radius: 5px;">
            <h3>Instrucciones</h3>
            <p><strong>Gracias por participar en esta encuesta. La misma es anónima y tiene fines estrictamente académicos para una tesis doctoral. Lea cuidadosamente y seleccione la opción que considere pertinente, al culminar presione Enviar</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # Mostrar datos demográficos
    datos_demograficos = mostrar_datos_demograficos()

    # Mostrar preguntas
    respuestas = mostrar_resumen()

    # Contador de respuestas
    num_respuestas = len(respuestas) - \
        len([v for v in respuestas.values() if v is None])
    total_preguntas = len(df)
    st.markdown(f"**Respuestas: {num_respuestas} / {total_preguntas}**")

    # Botón de envío
    if st.button("Enviar Encuesta"):
        if num_respuestas == total_preguntas:
            # Guardar las respuestas en la base de datos (Firebase o cualquier otra base)
            st.success("Gracias por completar la encuesta. ¡Globos y confites!")
            time.sleep(2)
            st.experimental_rerun()  # Recargar la página o redirigir a una página de finalización
        else:
            st.error("Por favor, responda todas las preguntas antes de enviar.")


# Ejecutar la aplicación
if __name__ == "__main__":
    app()
