import json
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128

class ConectorIA:
    def __init__(self,
                 ruta_modelo="models/modelo_final.keras",
                 ruta_clases="models/class_names.json"):

        self.ruta_modelo = Path(ruta_modelo)
        self.ruta_clases = Path(ruta_clases)

        # ----- Cargar modelo -----
        print("🧠 Cargando modelo IA desde:", self.ruta_modelo)
        self.model = load_model(self.ruta_modelo)

        # ----- Cargar nombres de clases -----
        print("📚 Cargando etiquetas desde:", self.ruta_clases)
        with open(self.ruta_clases, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        print("✔ IA lista para predecir banderas\n")

def predecir_bandera(self, ruta_imagen):
    """
    Recibe una ruta de imagen y devuelve la predicción de la IA:
    - nombre del país
    - probabilidad
    """
    try:
        # Cargar y preparar la imagen igual que en el entrenamiento
        img = image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0     # Normalizar igual que en el modelo
        img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión de batch

        # Realizar predicción
        pred = self.model.predict(img_array)[0]

        # Verificar que la predicción contiene valores válidos
        if np.all(np.isnan(pred)) or np.all(np.isinf(pred)):
            raise ValueError("La predicción contiene valores no válidos.")

        # Obtener el índice de la clase con la mayor probabilidad
        indice = np.argmax(pred)
        pais = self.class_names[indice]
        probabilidad = float(pred[indice])

        # Verificar si la probabilidad es válida
        if probabilidad < 0 or probabilidad > 1:
            raise ValueError("La probabilidad calculada es inválida.")

        return pais, probabilidad

    except Exception as e:
        print(f"Error al predecir la bandera: {e}")
        return "Error", 0.0

