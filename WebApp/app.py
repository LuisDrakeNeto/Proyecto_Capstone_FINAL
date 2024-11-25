import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

model = tf.keras.models.load_model('densenet201_model.h5')

def validar_y_renderizar(archivo):
    """Valida si el archivo subido es una imagen y retorna la imagen o un mensaje de error."""
    try:
        
        img = Image.open(archivo.name)
        img.load()  
        return img, None  
    except (UnidentifiedImageError, AttributeError, IOError):
        
        return None, f"""
        <div style="text-align: center; padding: 20px; background-color: #FF4136; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin-bottom: 10px;">Error</h2>
            <p style="color: white; font-size: 18px;">El archivo subido no es una imagen válida. Por favor, sube un archivo en formato PNG, JPEG, JPG, BMP o GIF.</p>
        </div>
        """

def clasificar_imagen(img):
    """Clasifica la imagen después de validar su formato."""
    if img is None:
        return """
        <div style="text-align: center; padding: 20px; background-color: #FF4136; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin-bottom: 10px;">Error</h2>
            <p style="color: white; font-size: 18px;">Por favor, sube una imagen para analizar.</p>
        </div>
        """
    try:
        # Procesar la imagen
        img = img.resize((224, 224))
        img = np.array(img)
        img = img.reshape((1, 224, 224, 3))
        img = img / 255.0
        prediccion = model.predict(img)

        clase_predicha = np.argmax(prediccion[0])

        # Mensajes según la clase predicha
        if clase_predicha in [2, 3]:
            mensaje = "🚨 URGENTE: Carie moderada o grave detectada"
            detalle = "Se recomienda una endodoncia inmediata. Visite a su dentista lo antes posible."
            color = "#FF4136"
        elif clase_predicha == 1:
            mensaje = "⚠️ Carie leve detectada"
            detalle = "Se recomienda un chequeo dental para tratar la carie antes de que empeore."
            color = "#FF851B"
        else:
            mensaje = "✅ No se detectaron caries"
            detalle = "¡Excelente! Mantenga su buena higiene bucal."
            color = "#2ECC40"

        return f"""
        <div style="text-align: center; padding: 20px; background-color: {color}; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin-bottom: 10px;">{mensaje}</h2>
            <p style="color: white; font-size: 18px;">{detalle}</p>
        </div>
        """
    except Exception as e:
        # Mensaje si ocurre un error durante el análisis
        return f"""
        <div style="text-align: center; padding: 20px; background-color: #FF4136; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin-bottom: 10px;">Error</h2>
            <p style="color: white; font-size: 18px;">Hubo un error al analizar la imagen. Por favor, intenta nuevamente.</p>
        </div>
        """

css = """
body {
    font-family: Arial, sans-serif;
    background-color: #2c3e50;
    color: white;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    background-color: #34495e;
}
.gr-button {
    background-color: #3498db !important;
    border: none !important;
    color: white !important;
}
.gr-button:hover {
    background-color: #2980b9 !important;
}
.gr-input, .gr-panel {
    border-color: #bdc3c7 !important;
    background-color: #2c3e50 !important;
    color: white !important;
}
.gr-form {
    background-color: #34495e !important;
    border-radius: 8px !important;
    padding: 15px !important;
}
.gr-box {
    border-radius: 8px !important;
    background-color: #2c3e50 !important;
}
.gr-padded {
    padding: 15px !important;
}
h1, h2, h3, p {
    color: white !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as interfaz:
    gr.Markdown(
    """
    # 🦷 Detección Avanzada de Caries Dentales con Inteligencia Artificial

    Sube una imagen de rayos X dental y nuestro modelo de IA analizará la presencia y gravedad de caries.
    """
    )
    with gr.Row():
        entrada_archivo = gr.File(label="Sube tu archivo aquí")
    with gr.Row():
        entrada_imagen = gr.Image(type="pil", label="Previsualización de la imagen", visible=True)
    with gr.Row():
        boton_clasificar = gr.Button("Analizar imagen", variant="primary")
        boton_limpiar = gr.Button("Limpiar", variant="secondary")
    with gr.Row():
        salida = gr.HTML(label="Resultado del análisis", elem_id="resultado-analisis")

    entrada_archivo.change(validar_y_renderizar, inputs=entrada_archivo, outputs=[entrada_imagen, salida])
    boton_clasificar.click(clasificar_imagen, inputs=entrada_imagen, outputs=salida)
    boton_limpiar.click(lambda: [None, None, ""], inputs=None, outputs=[entrada_archivo, entrada_imagen, salida])

    gr.Markdown(
    """
    ### Instrucciones:
    1. Sube una imagen clara de rayos X dental.
    2. Haz clic en "Analizar imagen".
    3. Revisa el resultado y las recomendaciones.
    4. Usa el botón "Limpiar" para resetear la entrada y el resultado.

    Recuerda: Esta herramienta es solo para referencia. Siempre consulta a un profesional dental para un diagnóstico preciso.
    """
    )

interfaz.launch()
