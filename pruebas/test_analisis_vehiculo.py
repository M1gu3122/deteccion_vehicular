"""Regresiones de color y asociacion sin inicializar pesos ni OCR."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analizar_vehiculo import asociar_placas
from color_vehiculo import estimar_color


class AnalisisVehiculoTests(unittest.TestCase):
    def test_colores_basicos(self):
        colores = {"rojo": (0, 0, 220), "azul": (220, 0, 0),
                   "blanco": (240, 240, 240), "negro": (20, 20, 20),
                   "gris/plateado": (120, 120, 120)}
        for nombre, bgr in colores.items():
            with self.subTest(color=nombre):
                imagen = np.full((100, 180, 3), bgr, dtype=np.uint8)
                self.assertEqual(estimar_color(imagen)["nombre"], nombre)

    def test_recorte_insuficiente(self):
        self.assertEqual(estimar_color(None)["nombre"], "indeterminado")
        self.assertEqual(estimar_color(np.zeros((10, 10, 3), np.uint8))["nombre"], "indeterminado")

    def test_placa_no_contamina_color(self):
        imagen = np.full((100, 180, 3), (0, 0, 220), dtype=np.uint8)
        imagen[48:76, 32:110] = (255, 255, 255)
        self.assertEqual(estimar_color(imagen, [(32, 48, 110, 76)])["nombre"], "rojo")

    def test_placas_no_se_comparten_entre_autos(self):
        autos = [{"caja": [0, 0, 200, 200]}, {"caja": [100, 100, 190, 190]}]
        placa = {"caja": [120, 140, 160, 160]}
        fuera = {"caja": [220, 220, 230, 230]}
        grupos = asociar_placas(autos, [placa, fuera])
        self.assertEqual(grupos, [[], [placa]])


if __name__ == "__main__":
    unittest.main()
