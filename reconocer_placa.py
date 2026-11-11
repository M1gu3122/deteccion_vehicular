import re
from pathlib import Path

import cv2
from ultralytics import YOLO

from lector_ocr import LectorOCR
from visor_imagen import mostrar


# Rutas de entrada
RUTA_IMAGEN = "imagenes/Screenshot 2026-11-02 141221.png"
RUTA_MODELO = "modelos/best.pt"

# Indice de clase de placa en el modelo
CLASE_PLACA = 0

# Confianza minima para aceptar una deteccion de placa
CONFIANZA_MINIMA = 0.40

# Tamano de inferencia (coincide con el export de OpenVINO)
TAMANO_INFERENCIA = 416


def cargar_modelo(ruta):
    """Carga la version OpenVINO si existe; si no, el .pt original."""
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="detect")
    return YOLO(str(ruta))


# Cargar los modelos una sola vez
modelo = cargar_modelo(RUTA_MODELO)
ocr = LectorOCR()


def normalizar_placa(texto):
    """
    Convierte el texto a mayusculas y conserva
    unicamente letras y numeros.
    """
    return LectorOCR.normalizar(texto)


def reconocer_placas(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen: {ruta_imagen}"
        )

    alto, ancho = imagen.shape[:2]

    # Detectar solo las placas (clase 0) con el modelo.
    resultados = modelo.predict(
        source=imagen,
        classes=[CLASE_PLACA],
        conf=CONFIANZA_MINIMA,
        imgsz=TAMANO_INFERENCIA,
        verbose=False,
    )

    placas_reconocidas = []

    for resultado in resultados:
        for caja in resultado.boxes or []:

            # Obtener las coordenadas de la placa
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
            confianza_yolo = float(caja.conf[0])

            # Anadir margen alrededor de la placa
            margen_x = int((x2 - x1) * 0.08)
            margen_y = int((y2 - y1) * 0.15)

            x1 = max(0, x1 - margen_x)
            y1 = max(0, y1 - margen_y)
            x2 = min(ancho, x2 + margen_x)
            y2 = min(alto, y2 + margen_y)

            recorte = imagen[y1:y2, x1:x2]

            if recorte.size == 0:
                continue

            # Reconocer el texto de la placa (modelos mobile + cache interno)
            lectura = ocr.leer(recorte)

            texto_original = lectura["texto_original"]
            texto_normalizado = normalizar_placa(texto_original) or lectura[
                "placa_normalizada"
            ]
            confianzas_ocr = lectura["confianzas_ocr"]

            datos = {
                "texto_original": texto_original,
                "placa_normalizada": texto_normalizado,
                "confianza_yolo": confianza_yolo,
                "confianzas_ocr": confianzas_ocr,
                "coordenadas": (x1, y1, x2, y2),
            }

            placas_reconocidas.append(datos)

            # Dibujar la deteccion en la imagen
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)

            etiqueta = texto_normalizado or "SIN LECTURA"

            # Etiqueta pegada al cuadro y confianza encima de ella
            y_etiqueta = max(25, y1 - 10)
            y_confianza = max(25, y_etiqueta - 35)

            cv2.putText(
                imagen,
                etiqueta,
                (x1, y_etiqueta),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                imagen,
                f"{confianza_yolo:.2f}",
                (x1, y_confianza),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            print("Resultado:", datos)

    # Guardar la imagen con las detecciones
    Path("resultados").mkdir(exist_ok=True)

    cv2.imwrite("resultados/imagen_procesada.jpg", imagen)

    mostrar("Reconocimiento de placas", imagen)

    return placas_reconocidas


if __name__ == "__main__":
    resultados = reconocer_placas(RUTA_IMAGEN)

    if not resultados:
        print("No se detectaron placas.")

    elif all(not r["placa_normalizada"] for r in resultados):
        print("Se detectaron placas, pero no se pudo leer su texto.")
