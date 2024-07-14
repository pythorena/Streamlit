import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('random_forest_regressor_default_42.sav')

# Título de la aplicación
st.title("Predicción de Diabetes")

# Descripción de la aplicación
st.write("""
Esta aplicación predice si una persona tiene diabetes en función de las siguientes características:
- Glucosa
- Presión Arterial
- Insulina
- Índice de Masa Corporal (BMI)
- Edad
- Función de Pedigrí de Diabetes
""")

# Recoger los datos del usuario
glucose = st.number_input("Nivel de glucosa", min_value=0)
blood_pressure = st.number_input("Presión Arterial", min_value=0)
insulin = st.number_input("Nivel de Insulina", min_value=0)
bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, format="%.2f")
age = st.number_input("Edad", min_value=0)
diabetes_pedigree_function = st.number_input("Función de Pedigrí de Diabetes", min_value=0.0, format="%.3f")

# Botón de predicción
if st.button("Predecir"):
    # Crear un array con los datos de entrada
    input_data = np.array([[glucose, blood_pressure, insulin, bmi, age, diabetes_pedigree_function]])

    # Hacer la predicción
    prediction = model.predict(input_data)

    # Mostrar el resultado
    if prediction[0] == 1:
        st.write("El modelo predice que esta persona tiene diabetes.")
    else:
        st.write("El modelo predice que esta persona no tiene diabetes.")