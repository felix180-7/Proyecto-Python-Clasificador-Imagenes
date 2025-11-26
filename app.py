import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

from conector_ia import ConectorIA


class VentanaBandera(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Identificar la imagen de la bandera")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet("background-color: #f0f0f5;")

        # Cargar la IA
        self.ia = ConectorIA()

        self.inicializarUI()

    def inicializarUI(self):
        # Layout principal vertical
        layout_principal = QVBoxLayout()
        layout_principal.setSpacing(20)

        # Label título
        self.label_titulo = QLabel("Identificar la imagen de la bandera")
        self.label_titulo.setAlignment(Qt.AlignCenter)
        self.label_titulo.setFont(QFont("Arial", 24, QFont.Bold))
        layout_principal.addWidget(self.label_titulo)

        # Label para imagen
        self.label_bandera = QLabel("Añade la imagen de una bandera")
        self.label_bandera.setAlignment(Qt.AlignCenter)
        self.label_bandera.setFixedSize(500, 350)
        self.label_bandera.setStyleSheet("""
            border: 2px solid #555;
            border-radius: 10px;
            background-color: #fff;
        """)

        # Centrado horizontal
        layout_imagen = QHBoxLayout()
        layout_imagen.addStretch(1)
        layout_imagen.addWidget(self.label_bandera)
        layout_imagen.addStretch(1)
        layout_principal.addLayout(layout_imagen)

        # Label de texto de resultado
        self.label_descripcion = QLabel("")
        self.label_descripcion.setAlignment(Qt.AlignCenter)
        self.label_descripcion.setFont(QFont("Arial", 16))
        layout_principal.addWidget(self.label_descripcion)

        # Layout botones
        layout_botones = QHBoxLayout()
        layout_botones.setSpacing(50)

        # Botón Cerrar
        self.boton_cerrar = QPushButton("Cerrar")
        self.boton_cerrar.setFixedSize(150, 50)
        self.boton_cerrar.setFont(QFont("Arial", 14, QFont.Bold))
        self.boton_cerrar.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.boton_cerrar.clicked.connect(self.cerrar_aplicacion)
        layout_botones.addWidget(self.boton_cerrar, alignment=Qt.AlignLeft)

        # Botón elegir imagen
        self.boton_elegir_imagen = QPushButton("Elegir imagen")
        self.boton_elegir_imagen.setFixedSize(150, 50)
        self.boton_elegir_imagen.setFont(QFont("Arial", 14, QFont.Bold))
        self.boton_elegir_imagen.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.boton_elegir_imagen.clicked.connect(self.elegir_imagen)
        layout_botones.addWidget(self.boton_elegir_imagen, alignment=Qt.AlignRight)

        layout_principal.addLayout(layout_botones)
        self.setLayout(layout_principal)

    def elegir_imagen(self):
        ruta_imagen, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona una imagen",
            "",
            "Archivos de imagen (*.png *.jpg *.bmp)"
        )

        if ruta_imagen:
            # Mostrar imagen seleccionada
            pixmap = QPixmap(ruta_imagen)
            pixmap = pixmap.scaled(self.label_bandera.size(),
                                   Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            self.label_bandera.setPixmap(pixmap)

            # --- Predicción con IA ---
            pais, prob = self.ia.predecir_bandera(ruta_imagen)

            self.label_descripcion.setText(
                f"Predicción: {pais} ({prob*100:.2f}% confianza)"
            )

    def cerrar_aplicacion(self):
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaBandera()
    ventana.show()
    sys.exit(app.exec_())
