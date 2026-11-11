import os
import re

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

from paddleocr import PaddleOCR


class LectorOCR:

    def __init__(self):
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )

    @staticmethod
    def normalizar(texto):
        texto = texto.upper()
        return re.sub(r"[^A-Z0-9]", "", texto)

    @staticmethod
    def _extraer_datos(resultado):
        datos = getattr(resultado, "json", resultado)
        if isinstance(datos, dict) and "res" in datos:
            datos = datos["res"]
        return datos if isinstance(datos, dict) else {}

    def leer(self, imagen_placa):
        resultados = self.ocr.predict(imagen_placa)

        textos = []
        confianzas = []

        for resultado in resultados:
            datos = self._extraer_datos(resultado)
            textos.extend(datos.get("rec_texts", []))
            confianzas.extend(datos.get("rec_scores", []))

        texto_original = "".join(textos).upper()
        texto_limpio = self.normalizar(texto_original)

        return {
            "texto_original": texto_original,
            "placa_normalizada": texto_limpio,
            "confianzas_ocr": confianzas,
        }
