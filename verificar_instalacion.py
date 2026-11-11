"""Comprueba dependencias y pesos locales sin descargar los modelos OCR."""

import sys
from pathlib import Path


def main():
    from importlib.metadata import version

    print(f"Python: {sys.version.split()[0]}")
    for paquete in ("ultralytics", "torch", "torchvision", "opencv-python",
                    "numpy", "openvino", "paddleocr", "paddlepaddle"):
        print(f"{paquete}: {version(paquete)}")

    import cv2
    import numpy as np
    from clasificador_vehiculo import ClasificadorVehiculo
    from detector import DetectorPlacas
    from lector_ocr import LectorOCR

    raiz = Path(__file__).resolve().parent
    for nombre in ("auto_cls_v2.pt", "best.pt"):
        ruta = raiz / "modelos" / nombre
        if not ruta.is_file() or ruta.stat().st_size < 1024:
            raise RuntimeError(f"Falta el modelo completo: {ruta}")
    clasificador = ClasificadorVehiculo()
    if not clasificador.disponible or len(clasificador.nombres) != 11:
        raise RuntimeError("El clasificador no tiene las 11 clases esperadas.")
    detector = DetectorPlacas()
    # Prueba de ejecucion, sin evaluar precision sobre una imagen vacia.
    imagen = np.zeros((224, 224, 3), dtype=np.uint8)
    clasificador.predecir(imagen)
    detector.detectar(imagen)
    assert LectorOCR.normalizar("FSN819GIRON") == "FSN819"
    print(f"OpenCV: {cv2.__version__}")
    print("OK: modelos cargados e inferencia ejecutada. OCR pendiente de su primera inicializacion.")


if __name__ == "__main__":
    main()
