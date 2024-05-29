import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Definimos las rutas de los datos
base_dir = 'Dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Verificar el número de imágenes
num_train_images = sum([len(files) for r, d, files in os.walk(train_dir)])
num_val_images = sum([len(files) for r, d, files in os.walk(validation_dir)])
num_test_images = sum([len(files) for r, d, files in os.walk(test_dir)])

print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_val_images}")
print(f"Number of test images: {num_test_images}")

# Crear generadores de datos
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Ajustar steps_per_epoch y validation_steps
steps_per_epoch = num_train_images // 20
validation_steps = num_val_images // 20
test_steps = num_test_images // 20

# Imprimir los valores para verificar
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")
print(f"Test steps: {test_steps}")

# Convertir a tf.data.Dataset y aplicar .repeat()
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None])
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None])
).repeat()

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 150, 150, 3], [None])
).repeat()

# Definir el modelo
def get_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

model = get_model((150, 150, 3))
model.summary()

# Compilar el modelo
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

compile_model(model)

# Entrenar el modelo
def train_model(model, train_dataset, validation_dataset, epochs=10):
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_dataset,
        validation_steps=validation_steps
    )
    return history

history = train_model(model, train_dataset, validation_dataset)
# Guardar el modelo entrenado
model.save('my_model.keras')
print("Modelo guardado como 'my_model.keras'.")

# Evaluar el modelo
def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps)
    return test_loss, test_accuracy

test_loss, test_accuracy = evaluate_model(model, test_dataset)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

def calculate_metrics(model, data_generator):
    # Obtener predicciones y etiquetas verdaderas
    predictions = []
    true_labels = []

    for i in range(validation_steps):
        batch_images, batch_labels = next(data_generator)
        batch_predictions = model.predict(batch_images)
        predictions.extend(batch_predictions.flatten())
        true_labels.extend(batch_labels)
    
    predictions = np.array(predictions) > 0.5
    true_labels = np.array(true_labels)

    # Calcular la matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    TN, FP, FN, TP = cm.ravel()

    return TP, TN, FP, FN

TP, TN, FP, FN = calculate_metrics(model, validation_generator)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


# Graficar las curvas de aprendizaje y otras métricas
def plot_learning_curves_and_metrics(history, TP, TN, FP, FN):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Graficar precisión y pérdida
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # Graficar TP, TN, FP, FN
    plt.subplot(3, 1, 3)
    metrics = [TP, TN, FP, FN]
    metric_labels = ['TP', 'TN', 'FP', 'FN']
    plt.bar(metric_labels, metrics, color=['blue', 'green', 'red', 'orange'])
    plt.title('Confusion Matrix Metrics')
    
    plt.tight_layout()
    plt.show()

plot_learning_curves_and_metrics(history, TP, TN, FP, FN)


# Predicciones del modelo
def model_predictions(model, test_generator):
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=-0.2)

    for i, (prediction, image, label) in enumerate(zip(predictions, test_images, test_labels)):
        axes[i, 0].imshow(image)
        axes[i, 0].get_xaxis().set_visible(False)
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 0].text(10., -1.5, f'Label {label}')
        axes[i, 1].bar([0, 1], prediction)
        axes[i, 1].set_xticks([0, 1])
        axes[i, 1].set_title(f"Prediction: {int(prediction[0] > 0.5)}")

    plt.show()

model_predictions(model, test_generator)
