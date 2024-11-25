import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1. Configuración general
# -------------------------------
batch_size = 32
img_size = (299, 299)  # InceptionV3 espera imágenes de 299x299
num_classes = 4  # Sin caries, leve, moderada, grave
epochs = 30  # Puedes empezar con 30 épocas
learning_rate = 1e-4

train_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Entrenamiento'
validation_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion'

# Crear carpeta para guardar los modelos
save_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/ModelosEntrenados/InceptionV3'
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
# 3. Definir el modelo InceptionV3 preentrenado
# -------------------------------
def build_inceptionv3_model():
    # Cargar el modelo preentrenado InceptionV3 sin las capas superiores
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Congelar las capas convolucionales del modelo preentrenado
    for layer in inception_base.layers:
        layer.trainable = False

    # Construir el modelo
    model = Sequential()

    # Agregar el modelo base InceptionV3
    model.add(inception_base)

    # Aplanar la salida
    model.add(Flatten())

    # Añadir capas densas personalizadas
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Regularización para evitar overfitting

    # Capa de salida con 4 neuronas (sin caries, leve, moderada, grave)
    model.add(Dense(num_classes, activation='softmax'))

    return model

# -------------------------------
# 4. Compilar y entrenar el modelo
# -------------------------------
model = build_inceptionv3_model()
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Implementar Early Stopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,  # 25 épocas para empezar
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# -------------------------------
# 5. Guardar el modelo entrenado
# -------------------------------
model_path = os.path.join(save_dir, 'inceptionv3_model.h5')
model.save(model_path)
print(f"Modelo InceptionV3 guardado en: {model_path}")