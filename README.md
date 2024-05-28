# Concrete Crack Classification
## TC3002B
### Renato Sebastian Ramírez Calva A01275401
## Descripción
El dataset seleccionado para el modelo consta de 40,000 imágenes en total, distribuido en 20k de concreto quebrado y 20k de concreto no quebrado. Estas imagenes fueron tomadas de varios edificio de campus de METU(Universidad Técnica de Medio Oriente) para su clasificación.
## Fuente
Las imágenes fueron adquiridas del dataset titulado [Concrete Crack Images for Classification de Kaggle](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification?select=Negative). 
Este conjunto fue seleccionado por la cantidad de imagenes y su formato en su haber.

El dataset fue creado originalmente por el usuario ArnavR. El formato de las imagenes es jpg y todas cumplen con 277 x 277 pixeles.


## Dataset
El dataset original no está incluido en este repositorio. Sin embargo, se puede acceder desde este enlace de [Google Drive](https://drive.google.com/drive/folders/1esR6ZMOZ3Ljps-FKKPdS_Y1fAdb_qZ6E?ths=true).

Se dividieron las imagenes en dos carpetas: train, test. La carpeta train contiene 32000 imágenes, mientras que la carpeta test contiene 8000 imágenes. 

Las imagenes fueron renombradas acorde a su clasepara no tener problemas de uso. 

En su total hay 2 clases para las imagenes, POSITIVE Y NEGATIVE, POSITIVE refiere a imagenes donde se presenta una abertura de concreto, NEGATIVE apunta a imagenes donde el concreto esta intacto.

Se uso el prefijo 0000xT, donde x es el numero de la imagen y T el termino referente a la clase, ej: 00001P.jpg.

## Articulos Guia
Para la realizacion de este modelo se usaroon varios articulos como guia para idealizar el funcionamiento, cabe recalcar que varios proyectos mencionados en los articulos, usan modelos pre-entrenados.

El modelo realizado en este repositorio fue hecho desde cero para la comprension de las redes neuronales, aunque sea a un nivel superficial.

[The Effectiveness of Data Augmentation in Image Classification using Deep Learning by Jason Wang, Luis Perez](https://arxiv.org/pdf/1712.04621).

[When Residual Learning Meets Dense Aggregation: Rethinking the Aggregation of Deep Neural Networks by Zhiyu Zhu, Zhen-Peng Bian,Junhui Hou1, Yi Wang and Lap-Pui Chau](https://arxiv.org/pdf/2004.08796).

[Going deeper with convolutions by Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich](https://arxiv.org/pdf/1409.4842).

[ImageNet Classification with Deep Convolutional Neural Networks by Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## Resultados primera evaluacion
![Train and Validation de Modelo Sin Refinar](./img/Accuracy.png)



En la grafica la precisión del entrenamiento comienza alrededor del 98% y aumenta constantemente hasta más del 99,5%. 

La precisión de la validación comienza en alrededor del 97% y aumenta hasta alrededor del 99%. 

Esto sugiere que el modelo está funcionando bien y no está sobreajustado. Sin embargo, monitorear de cerca la precisión de la validación durante proximos entrenamiento seria buena idea para asegurarse de que el modelo no se sobreajuste.

![Epochs](./img/AccuracyModel.png)

En general, la imagen muestra que el modelo se ha entrenado con éxito. 

La precisión de entrenamiento y la precisión de validación son ambas altas, y la pérdida de entrenamiento y la pérdida de validación son ambas bajas. 

Esto sugiere que el modelo es capaz de aprender de los datos de entrenamiento y generalizarse bien a los datos nuevos.

## Conclusiones 

La precisión de validación también aumenta a lo largo de las 10 épocas, pero no tan rápidamente como la precisión de entrenamiento. 

Esto es un buen signo, ya que sugiere que el modelo no está sobreajustándose a los datos de entrenamiento.

La pérdida de entrenamiento disminuye constantemente a lo largo de las 10 épocas. Esto es un buen signo, ya que sugiere que el modelo está aprendiendo a cometer menos errores.

En general, el modelo se ha entrenado con éxito. Sin embargo, estos son sólo los resultados del entrenamiento del modelo. 

Es importante evaluar el rendimiento del modelo en un conjunto de datos de prueba independiente para asegurarse de que se adapta bien a los datos nuevos.


