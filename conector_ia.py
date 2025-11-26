import json
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128


class ConectorIA:
    def __init__(self,
                 ruta_modelo="models/modelo_final",
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

        # Cargar y preparar imagen igual que en el entrenamiento
        img = image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0     # Normalizar igual que en el modelo
        img_array = np.expand_dims(img_array, axis=0)

        # Realizar predicción
        pred = self.model.predict(img_array)[0]

        indice = np.argmax(pred)
        pais = self.class_names[indice]
        probabilidad = float(pred[indice])

        return pais, probabilidad
