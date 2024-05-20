import os

def renombrar_imagenes(carpeta):
  """
  Renombra y renumera las imágenes en una carpeta especificada.

  Args:
    carpeta: Ruta de la carpeta que contiene las imágenes.
  """
  contador = 1
# Cambia N.jpg O P.jpg
  for filename in os.listdir(carpeta):
    if filename.endswith(".jpg"):
      nuevo_nombre = f"{str(contador).zfill(5)}N.jpg"
    #   print(nuevo_nombre)
      os.rename(os.path.join(carpeta, filename), os.path.join(carpeta, nuevo_nombre))
      contador += 1

# Renombrar imágenes en la carpeta "POSITIVE"
# renombrar_imagenes("Dataset\Positive")

# Renombrar imágenes en la carpeta "NEGATIVE"
renombrar_imagenes("Dataset\Negative")
 