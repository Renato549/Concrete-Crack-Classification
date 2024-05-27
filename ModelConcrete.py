import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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

# Evaluar el modelo
def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps)
    return test_loss, test_accuracy

test_loss, test_accuracy = evaluate_model(model, test_dataset)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Graficar las curvas de aprendizaje
def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 9))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

plot_learning_curves(history)

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
