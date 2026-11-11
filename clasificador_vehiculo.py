from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"

# Modelo de clasificacion marca/modelo (opcional).
MODELO_CLASIFICADOR = MODELOS_DIR / "auto_cls_v2.pt"

# Tamano de inferencia (coincide con el export de OpenVINO).
TAMANO_INFERENCIA = 224

# Ancho minimo del recorte para intentar clasificar (autos lejanos no sirven).
ANCHO_MINIMO = 100

# Confianza minima para aceptar una prediccion.
CONFIANZA_MINIMA = 0.45


def cargar_modelo(ruta):
    """Carga la version OpenVINO si existe; si no, el .pt original."""
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="classify")
    return YOLO(str(ruta), task="classify")


class ClasificadorVehiculo:
    """Clasificador de marca/modelo opcional.

    Si el modelo no existe, ``disponible`` es False y ``predecir`` devuelve
    None; el resto del pipeline sigue funcionando sin cambios.
    """

    def __init__(
        self,
        ruta=MODELO_CLASIFICADOR,
        tamano_inferencia=TAMANO_INFERENCIA,
        ancho_minimo=ANCHO_MINIMO,
        confianza_minima=CONFIANZA_MINIMA,
    ):
        self.ruta = Path(ruta)
        self.tamano_inferencia = tamano_inferencia
        self.ancho_minimo = ancho_minimo
        self.confianza_minima = confianza_minima
        self.modelo = None
        self.nombres = {}

        if not self.ruta.exists():
            return

        try:
            self.modelo = cargar_modelo(self.ruta)
            self.nombres = self.modelo.names
        except Exception as error:
            print(f"No se pudo cargar el clasificador: {error}")
            self.modelo = None

    @property
    def disponible(self):
        return self.modelo is not None

    def predecir(self, recorte):
        """Devuelve (etiqueta, confianza) o None si no aplica."""
        if not self.disponible or recorte is None or recorte.size == 0:
            return None

        if recorte.shape[1] < self.ancho_minimo:
            return None

        resultado = self.modelo.predict(
            recorte,
            imgsz=self.tamano_inferencia,
            verbose=False,
        )[0]

        if resultado.probs is None:
            return None

        indice = int(resultado.probs.top1)
        confianza = float(resultado.probs.top1conf)

        if confianza < self.confianza_minima:
            return None

        return self.nombres[indice], confianza
