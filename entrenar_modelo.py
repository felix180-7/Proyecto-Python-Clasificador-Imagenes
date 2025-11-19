"""
train_cnn.py
Entrena una red neuronal que reconoce banderas europeas.

USO:
    python train_cnn.py

Requisitos:
- Tener las imágenes en data/train/<pais>
- Tener 10 imágenes por país
- Haber instalado: tensorflow, pillow, numpy, matplotlib
"""

import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------------------
# CONFIGURACIÓN BÁSICA (NO TOCAR)
# -------------------------------------

IMG_SIZE = 128               # Tamaño al que reducimos las imágenes
BATCH_SIZE = 16
EPOCHS = 40                  # Cuántas vueltas da la IA para aprender

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos" / "entrenamiento"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------
# CARGA DE IMÁGENES DESDE CARPETAS
# -------------------------------------

print("🔄 Cargando imágenes desde:", DATA_DIR)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("🟩 Países detectados:", class_names)
print("Total países:", num_classes)

# Aceleramos el entrenamiento un poquito
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -------------------------------------
# CREAMOS LA RED NEURONAL (CNN)
# -------------------------------------

def crear_modelo():
    modelo = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo

model = crear_modelo()
model.summary()

# -------------------------------------
# ENTRENAR EL MODELO
# -------------------------------------

print("🚀 Entrenando la IA...")

hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("✔ Entrenamiento terminado")

# -------------------------------------
# GUARDAR MODELO Y CLASES
# -------------------------------------

model.save(MODELS_DIR / "modelo_final")
print("💾 Modelo guardado en /models/modelo_final")

with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2, ensure_ascii=False)

print("📚 class_names.json guardado")

# -------------------------------------
# GUARDAR GRÁFICA DE ENTRENAMIENTO
# -------------------------------------

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(hist.history["accuracy"], label="Entrenamiento")
plt.plot(hist.history["val_accuracy"], label="Validación")
plt.title("Precisión")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.history["loss"], label="Entrenamiento")
plt.plot(hist.history["val_loss"], label="Validación")
plt.title("Pérdida")
plt.legend()

plt.tight_layout()
plt.savefig(MODELS_DIR / "grafica_entrenamiento.png")

print("📈 Gráfica guardada en grafica_entrenamiento.png")
print("🎉 TODO LISTO — MODELO ENTRENADO")
