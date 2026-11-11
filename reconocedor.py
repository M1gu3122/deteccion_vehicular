import cv2

from config import IMAGEN_SALIDA, RESULTADOS_DIR, MARGEN_HORIZONTAL, MARGEN_VERTICAL

from detector import DetectorPlacas
from lector_ocr import LectorOCR
from visor_imagen import mostrar


class ReconocedorPlacas:

    def __init__(self):
        self.detector = DetectorPlacas()
        self.lector = LectorOCR()

    def procesar_imagen(self, ruta_imagen):
        imagen = cv2.imread(str(ruta_imagen))

        if imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")

        alto, ancho = imagen.shape[:2]

        detecciones = self.detector.detectar(imagen)
        resultados_finales = []

        for deteccion in detecciones:
            x1, y1, x2, y2 = deteccion["coordenadas"]

            # Calcular el margen del recorte
            margen_x = int((x2 - x1) * MARGEN_HORIZONTAL)
            margen_y = int((y2 - y1) * MARGEN_VERTICAL)

            # Evitar que el recorte salga de la imagen
            x1r = max(0, x1 - margen_x)
            y1r = max(0, y1 - margen_y)
            x2r = min(ancho, x2 + margen_x)
            y2r = min(alto, y2 + margen_y)

            placa = imagen[y1r:y2r, x1r:x2r]

            if placa.size == 0:
                continue

            # Reconocer caracteres
            lectura = self.lector.leer(placa)

            resultado = {
                "placa": lectura["placa_normalizada"],
                "texto_original": lectura["texto_original"],
                "confianza_yolo": deteccion["confianza"],
                "confianzas_ocr": lectura["confianzas_ocr"],
                "coordenadas": (x1, y1, x2, y2),
            }

            resultados_finales.append(resultado)

            # Dibujar el rectángulo y el texto
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

            etiqueta = resultado["placa"] or "SIN LECTURA"

            cv2.putText(
                imagen,
                etiqueta,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Crear el directorio si no existe
        RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(IMAGEN_SALIDA), imagen):
            raise OSError(f"No se pudo guardar la imagen: {IMAGEN_SALIDA}")

        # Mostrar la imagen procesada con zoom y arrastre.
        mostrar("Reconocimiento de placas", imagen)

        return resultados_finales
