from pathlib import Path

from ultralytics import YOLO

from config import MODELO_PLACAS


# Indice de clase de placa en el modelo
CLASE_PLACA = 0

# Confianza minima para aceptar una deteccion de placa
CONFIANZA_PLACA = 0.40

# Tamano de inferencia (coincide con el export de OpenVINO)
TAMANO_INFERENCIA = 416


def cargar_modelo(ruta):
    """Carga la version OpenVINO si existe; si no, el .pt original."""
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="detect")
    return YOLO(str(ruta))


class DetectorPlacas:

    def __init__(self):
        if not MODELO_PLACAS.exists():
            raise FileNotFoundError(
                f"No se encontro el modelo: {MODELO_PLACAS}"
            )

        self.modelo = cargar_modelo(MODELO_PLACAS)

    def detectar(self, imagen):
        resultados = self.modelo.predict(
            source=imagen,
            classes=[CLASE_PLACA],
            conf=CONFIANZA_PLACA,
            imgsz=TAMANO_INFERENCIA,
            verbose=False,
        )

        detecciones = []

        for resultado in resultados:
            if resultado.boxes is None:
                continue

            for caja in resultado.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

                detecciones.append({
                    "coordenadas": (x1, y1, x2, y2),
                    "confianza": float(caja.conf[0]),
                })

        return detecciones
