import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# 1. Verificar y cargar imágenes
# -------------------------------
def verificar_imagenes(ruta):
    imagenes_validas = []
    for archivo in os.listdir(ruta):
        archivo_path = os.path.join(ruta, archivo)
        try:
            with Image.open(archivo_path) as img:
                img.verify()  # Verificar si el archivo está dañado
                imagenes_validas.append(archivo)
        except (IOError, SyntaxError) as e:
            print(f"Error con la imagen {archivo}: {e}")
    return imagenes_validas

# -------------------------------
# 2. Reducción de ruido
# -------------------------------
def reducir_ruido(imagen_cv2):
    # Aplicar un filtro Gaussiano para suavizar la imagen
    return cv2.GaussianBlur(imagen_cv2, (5, 5), 0)

# -------------------------------
# 3. Redimensionar imágenes
# -------------------------------
def redimensionar_imagen(imagen, nuevo_tamano=(128, 128)):
    return imagen.resize(nuevo_tamano)

# -------------------------------
# 4. Normalización de imágenes
# -------------------------------
def normalizar_imagen(imagen):
    imagen_array = np.array(imagen) / 255.0  # Escalar los valores a rango 0-1
    return imagen_array

# -------------------------------
# 5. Ajustar brillo y contraste
# -------------------------------
def ajustar_brillo_contraste(imagen, factor_brillo=1.3, factor_contraste=1.5):
    enhancer_brightness = ImageEnhance.Brightness(imagen)
    imagen_brillo = enhancer_brightness.enhance(factor_brillo)
    
    enhancer_contrast = ImageEnhance.Contrast(imagen_brillo)
    imagen_contraste = enhancer_contrast.enhance(factor_contraste)
    
    return imagen_contraste

# -------------------------------
# 6. Aumento de datos
# -------------------------------
def aumento_de_datos(imagen_array, num_imagenes_aumentadas=5):
    datagen = ImageDataGenerator(
        rotation_range=40,        # Rotar hasta 40 grados
        width_shift_range=0.2,    # Desplazamiento horizontal
        height_shift_range=0.2,   # Desplazamiento vertical
        shear_range=0.2,          # Corte
        zoom_range=0.2,           # Zoom
        brightness_range=[0.8, 1.2],  # Rango de brillo aleatorio
        horizontal_flip=True,     # Volteo horizontal
        vertical_flip=True,       # Volteo vertical
        fill_mode='nearest'       # Modo de relleno
    )
    
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Expandimos dimensiones
    data_gen = datagen.flow(imagen_array, batch_size=1)  # Generar imagen aumentada
    
    imagenes_aumentadas = []
    for _ in range(num_imagenes_aumentadas):
        batch = next(data_gen)
        imagen_aumentada = batch[0].astype(np.uint8)  # Convertir a formato de imagen
        imagenes_aumentadas.append(imagen_aumentada)
    
    return imagenes_aumentadas

# -------------------------------
# 7. Procesar todas las imágenes
# -------------------------------
def procesar_imagenes(ruta_imagenes, ruta_guardado, clase, num_imagenes_aumentadas=5):
    imagenes_validas = verificar_imagenes(ruta_imagenes)
    
    # Crear subcarpeta para las imágenes procesadas de esta clase (leve, moderada, grave, sin_caries)
    ruta_clase_guardado = os.path.join(ruta_guardado, clase)
    os.makedirs(ruta_clase_guardado, exist_ok=True)
    
    for archivo in imagenes_validas:
        try:
            archivo_path = os.path.join(ruta_imagenes, archivo)
            img = Image.open(archivo_path)
            img_cv2 = cv2.imread(archivo_path)
            
            # 1. Reducir ruido
            img_cv2_sin_ruido = reducir_ruido(img_cv2)
            
            # 2. Convertir a PIL Image y redimensionar
            img_pil = Image.fromarray(img_cv2_sin_ruido)
            img_redimensionada = redimensionar_imagen(img_pil)
            
            # 3. Ajustar brillo y contraste
            img_contraste = ajustar_brillo_contraste(img_redimensionada)
            
            # 4. Normalizar imagen
            img_normalizada = normalizar_imagen(img_contraste)
            
            # Guardar la imagen normalizada (transformada)
            ruta_imagen_guardada = os.path.join(ruta_clase_guardado, f"procesada_{archivo}")
            Image.fromarray((img_normalizada * 255).astype(np.uint8)).save(ruta_imagen_guardada)
            print(f"Imagen procesada y guardada: {ruta_imagen_guardada}")
            
            # 5. Aumento de datos
            img_aumentada_array = np.array(img_contraste)
            imagenes_aumentadas = aumento_de_datos(img_aumentada_array, num_imagenes_aumentadas)
            
            # Guardar imágenes aumentadas
            for i, imagen_aumentada in enumerate(imagenes_aumentadas):
                ruta_aumento = os.path.join(ruta_clase_guardado, f"augmentada_{i}_{archivo}")
                Image.fromarray(imagen_aumentada).save(ruta_aumento)
                print(f"Imagen aumentada y guardada: {ruta_aumento}")
        
        except Exception as e:
            print(f"Error procesando la imagen {archivo}: {e}")

# -------------------------------
# Configurar rutas y ejecutar
# -------------------------------
if __name__ == "__main__":
    ruta_carie_grave = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\Muestras\Carie_Grave'
    ruta_carie_leve = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\Muestras\Carie_Leve'
    ruta_carie_moderada = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\Muestras\Carie_Moderada'
    ruta_sin_caries = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\Muestras\Sin_Caries'
    
    ruta_guardado = r'C:\Users\USER\Documents\UPN 10 CICLO\CAPSTONE PROJECT\PROYECTO PYTHON\Entrenamiento'
    
    # Procesar imágenes de cada clase y aplicar aumento de datos
    procesar_imagenes(ruta_carie_grave, ruta_guardado, 'Carie_Grave', num_imagenes_aumentadas=5)
    procesar_imagenes(ruta_carie_leve, ruta_guardado, 'Carie_Leve', num_imagenes_aumentadas=5)
    procesar_imagenes(ruta_carie_moderada, ruta_guardado, 'Carie_Moderada', num_imagenes_aumentadas=5)
    procesar_imagenes(ruta_sin_caries, ruta_guardado, 'Sin_Caries', num_imagenes_aumentadas=5)
