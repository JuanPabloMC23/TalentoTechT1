from matplotlib import pyplot as plt
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 

#Cargar modelo entrenado
model = tf.keras.models.load_model(r'D:\Documentos\JPMC\Cursos\Bootcamp IA intermedio\mlp_model.h5')

#Función para preprocesar la imagen
def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255
    image = image.reshape(1, 28*28)
    return image

#Título de la aplicación

st.title('Clasificación de dígitos manuscritos')
uploaded_file = st.file_uploader('Cargar una imagen de un digito(0-9)', type=('png', 'jpg','jpeg'))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)
    preprocessed_image = preprocess_image(image)
    
    #Hacer la predicción
    prediction = model.predict(preprocessed_image)
    predicted_digit = np.argmax(prediction)
    
    #Mostrar la predicción
    st.write(f'Predicción: **{predicted_digit}**')
    
    for i in range(10):
        #st.write(f'Dígito {i}: {prediction[0][i]:.4f}')
        st.write(f"Dígito {i}: {prediction[0][i]:.4f}")

plt.imshow(preprocessed_image.reshape(28, 28), cmap='gray')  # Convierte a 28x28 para visualizar
plt.axis('off')
st.pyplot(plt)    
    