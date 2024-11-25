import os
import shutil
import random

# Definir rutas
train_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Entrenamiento'
validation_dir = r'C:/Users/USER/Documents/UPN 10 CICLO/CAPSTONE PROJECT/PROYECTO PYTHON/Validacion'

# Crear carpetas de validación si no existen
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# Porcentaje de datos a mover a validación (30%)
validation_split = 0.3

# Listar las clases (subcarpetas)
clases = os.listdir(train_dir)

for clase in clases:
    # Crear carpeta de validación para la clase si no existe
    clase_train_dir = os.path.join(train_dir, clase)
    clase_val_dir = os.path.join(validation_dir, clase)
    
    # Crear la carpeta de validación correspondiente si no existe
    if not os.path.exists(clase_val_dir):
        os.makedirs(clase_val_dir)

    # Listar todas las imágenes de la clase
    imagenes = os.listdir(clase_train_dir)
    
    # Mezclar imágenes aleatoriamente
    random.shuffle(imagenes)
    
    # Calcular cuántas mover a la carpeta de validación (30% del total)
    num_val = int(len(imagenes) * validation_split)
    
    # Mover imágenes a la carpeta de validación
    for i in range(num_val):
        img = imagenes[i]
        src = os.path.join(clase_train_dir, img)
        dst = os.path.join(clase_val_dir, img)
        shutil.move(src, dst)

    print(f"Movidas {num_val} imágenes de la clase {clase} a validación.")
