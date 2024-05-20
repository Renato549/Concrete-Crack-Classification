# Concrete Crack Classification
## TC3002B
### Renato Sebastian Ramírez Calva A01275401
## Descripción
El dataset seleccionado para el modelo consta de 40,000 imágenes en total, distribuido en 20k de concreto quebrado y 20k de concreto no quebrado. Estas imagenes fueron tomadas de varios edificio de campus de METU para su clasificación
## Fuente
Las imágenes fueron adquiridas del dataset titulado [Concrete Crack Images for Classification de Kaggle](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification?select=Negative). 
Este conjunto fue seleccionado por la cantidad de imagenes y su formato en su haber.

El dataset fue creado originalmente por el usuario ArnavR. El formato de las imagenes es jpg y todas cumplen con 277 x 277 pixeles.
Data augmentation no fue aplicado en las imagenes para voltear o rotar en el set original

## Dataset
El dataset original no está incluido en este repositorio. Sin embargo, se puede acceder desde este enlace de [Google Drive](https://drive.google.com/drive/folders/1esR6ZMOZ3Ljps-FKKPdS_Y1fAdb_qZ6E?ths=true).

Se dividieron las imagenes en dos carpetas: train, test. La carpeta train contiene 32000 imágenes, mientras que la carpeta test contiene 8000 imágenes, las imagenes fueron renombradas acorde a su clasepara no tener problemas de uso. 
En su total hay 2 clases para las imagenes, POSITIVE Y NEGATIVE, POSITIVE refiere a imagenes donde se presenta una abertura de concreto, NEGATIVE apunta a imagenes donde el concreto esta intacto.
Se uso el prefijo 0000xT, donde x es el numero de la imagen y T el termino referente a la clase, ej: 00001P.jpg.



