"""Regresiones de extraccion de matricula sin inicializar modelos OCR."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lector_ocr import LectorOCR


class LectorOCRTests(unittest.TestCase):
    def test_descarta_ciudad(self):
        for texto in ("FSN819GIRON", "FSN 819 GIRON", "fsn-819", "GIRON FSN819"):
            with self.subTest(texto=texto):
                self.assertEqual(LectorOCR.normalizar(texto), "FSN819")

    def test_no_acepta_lecturas_incompletas_o_ambiguas(self):
        for texto in ("GIRON", "FSN81", "FSN8190", "FSN819 ABC123", "30A60434"):
            with self.subTest(texto=texto):
                self.assertEqual(LectorOCR.normalizar(texto), "")

    def test_ocr_conserva_texto_original_y_separa_bloques(self):
        class OCRSimulado:
            def predict(self, imagen):
                return [{"res": {"rec_texts": ["GIRON", "FSN", "819"],
                                  "rec_scores": [.99, .98, .97]}}]
        lector = LectorOCR.__new__(LectorOCR)
        lector.ocr = OCRSimulado()
        resultado = lector.leer(None)
        self.assertEqual(resultado["placa_normalizada"], "FSN819")
        self.assertEqual(resultado["texto_original"], "GIRONFSN819")
        self.assertEqual(len(resultado["confianzas_ocr"]), 3)


if __name__ == "__main__":
    unittest.main()
