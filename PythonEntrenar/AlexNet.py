import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# 1. Configuración general
# -------------------------------
batch_size = 32
img_size = (227, 227)  # AlexNet espera imágenes de 227x227
num_classes = 4  # Sin caries, leve, moderada, grave
epochs = 30  # Puedes empezar con 30 épocas
learning_rate = 1e-4

train_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Entrenamiento'
validation_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion'

# Crear carpeta para guardar los modelos
save_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/ModelosEntrenados/AlexNet'
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
# 3. Definir el modelo AlexNet
# -------------------------------
def build_alexnet_model():
    model = Sequential()

    # Capa 1: Conv -> ReLU -> MaxPooling -> BatchNormalization
    model.add(Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)))
    model.add(MaxPooling2D((3, 3), strides=2))
    model.add(BatchNormalization())

    # Capa 2: Conv -> ReLU -> MaxPooling -> BatchNormalization
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=2))
    model.add(BatchNormalization())

    # Capa 3: Conv -> ReLU -> BatchNormalization
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Capa 4: Conv -> ReLU -> BatchNormalization
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # Capa 5: Conv -> ReLU -> MaxPooling -> BatchNormalization
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=2))
    model.add(BatchNormalization())

    # Aplanar las capas convolucionales
    model.add(Flatten())

    # Capa 6: Capa completamente conectada -> ReLU -> Dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Capa 7: Capa completamente conectada -> ReLU -> Dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Capa 8: Capa de salida -> Softmax
    model.add(Dense(num_classes, activation='softmax'))

    return model

# -------------------------------
# 4. Compilar y entrenar el modelo
# -------------------------------
model = build_alexnet_model()
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Implementar Early Stopping para evitar sobreentrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# -------------------------------
# 5. Guardar el modelo entrenado
# -------------------------------
model_path = os.path.join(save_dir, 'alexnet_model.h5')
model.save(model_path)
print(f"Modelo AlexNet guardado en: {model_path}")
