import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Cargar el modelo guardado
model = tf.keras.models.load_model('my_model.keras')
print("Modelo cargado exitosamente.")

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = 'D:\\Renato\\Desktop\\Dataset\\ImgTest'

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(image_path, target_size=(150, 150)):
  img = Image.open(image_path).convert('RGB')
  img = img.resize(target_size)
  img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  
  img_array = np.expand_dims(img_array, axis=0)  
  return img_array

# Recorrer la carpeta y cargar las imágenes
imagenes = []
for filename in os.listdir(carpeta_imagenes):
  if filename.endswith('.png'):
    image_path = os.path.join(carpeta_imagenes, filename)
    imagenes.append(load_and_preprocess_image(image_path))

# Procesar y predecir cada imagen
for i, new_image in enumerate(imagenes):
  # Hacer la predicción
  prediction = model.predict(new_image)
  print("Hola")

  # Obtener el nombre de la imagen
  nombre_imagen = os.path.join(carpeta_imagenes, os.listdir(carpeta_imagenes)[i])

  # Mostrar la imagen y la predicción
  plt.figure()
  plt.imshow(new_image[0])
  plt.axis('off')
  plt.title(f"Prediction: {int(prediction[0] > 0.5)}")
  plt.show()

  # Imprimir la predicción en consola
  print(f"Imagen: {nombre_imagen} - Predicción: {prediction[0][0]}")

