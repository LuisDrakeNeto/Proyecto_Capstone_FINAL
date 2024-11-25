import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Configuración general
# -------------------------------
batch_size = 32
img_size = (128, 128)
num_classes = 4  # Sin caries, leve, moderada, grave
epochs = 10
learning_rate = 1e-4

train_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Entrenamiento'
validation_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion'

# Crear carpeta para guardar los modelos
save_dir = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\ModelosEntrenados\CNN_Standard'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# -------------------------------
# 2. Preprocesamiento y aumento de datos
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# -------------------------------
# 3. Definir el modelo CNN
# -------------------------------
def build_cnn_model():
    model = Sequential()

    # Capas convolucionales + pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Aplanar la salida y pasarla a una capa densa
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Regularización para evitar overfitting

    # Capa de salida con 4 neuronas (4 clases: sin caries, leve, moderada, grave)
    model.add(Dense(num_classes, activation='softmax'))

    return model

# -------------------------------
# 4. Compilar y entrenar el modelo
# -------------------------------
model = build_cnn_model()
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# -------------------------------
# 5. Guardar el modelo entrenado
# -------------------------------
model_path = os.path.join(save_dir, 'cnn_model.h5')
model.save(model_path)
print(f"Modelo CNN estándar guardado en: {model_path}")
