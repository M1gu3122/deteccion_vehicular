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
        """Extrae una matricula de auto de tres letras y tres numeros.

        Descarta ciudad y otras leyendas. No inventa caracteres si la
        lectura es incompleta o contiene dos matriculas distintas.
        """
        candidatos = re.findall(
            r"(?<![A-Z0-9])[A-Z]{3}[\s.-]*[0-9]{3}(?![0-9])",
            texto.upper(),
        )
        placas = {re.sub(r"[^A-Z0-9]", "", candidato)
                  for candidato in candidatos}
        return placas.pop() if len(placas) == 1 else ""

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
        # Mantener los limites de bloques evita unir ciudad y matricula.
        texto_limpio = self.normalizar(" ".join(textos))

        return {
            "texto_original": texto_original,
            "placa_normalizada": texto_limpio,
            "confianzas_ocr": confianzas,
        }
