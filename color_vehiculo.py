"""Estimacion heuristica del color de carroceria, sin pesos adicionales."""

import cv2
import numpy as np


def estimar_color(recorte, placas=()):
    """Devuelve color y proporcion dominante; no es probabilidad del modelo.

    Muestrea bandas centrales/bajas para reducir cielo, ventanas y ruedas.
    Las placas son cajas locales al recorte. Gris y plateado se agrupan:
    una imagen RGB no permite distinguir de forma fiable el acabado metalico.
    """
    desconocido = {"nombre": "indeterminado", "proporcion": 0.0}
    if recorte is None or recorte.size == 0:
        return desconocido
    alto, ancho = recorte.shape[:2]
    if min(alto, ancho) < 30:
        return desconocido
    mascara = np.zeros((alto, ancho), np.uint8)
    mascara[int(alto * .48):int(alto * .76),
            int(ancho * .18):int(ancho * .82)] = 255
    for x1, y1, x2, y2 in placas:
        cv2.rectangle(mascara, (max(0, x1 - 5), max(0, y1 - 5)),
                      (min(ancho - 1, x2 + 5), min(alto - 1, y2 + 5)), 0, -1)
    imagen = cv2.resize(recorte, (160, 160), interpolation=cv2.INTER_AREA)
    mascara = cv2.resize(mascara, (160, 160), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(cv2.GaussianBlur(imagen, (5, 5), 0), cv2.COLOR_BGR2HSV)
    pixeles = hsv[mascara > 0]
    if len(pixeles) < 100:
        return desconocido
    h, s, v = pixeles.T
    etiquetas = np.full(len(h), "gris/plateado", dtype="U20")
    etiquetas[v < 55] = "negro"
    etiquetas[(s < 35) & (v > 200)] = "blanco"
    cromatico = (s >= 55) & (v >= 55)
    rangos = (("rojo", (h < 10) | (h >= 170)),
              ("naranja", (h >= 10) & (h < 22)),
              ("amarillo", (h >= 22) & (h < 35)),
              ("verde", (h >= 35) & (h < 85)),
              ("azul", (h >= 85) & (h < 135)),
              ("morado", (h >= 135) & (h < 170)))
    for nombre, seleccion in rangos:
        etiquetas[cromatico & seleccion] = nombre
    etiquetas[cromatico & (h >= 10) & (h < 30) & (v < 160)] = "marron"
    etiquetas[(h >= 10) & (h < 35) & (s >= 35) & (s < 100) & (v >= 160)] = "beige"
    nombres, conteos = np.unique(etiquetas, return_counts=True)
    indice = int(conteos.argmax())
    proporcion = float(conteos[indice] / len(etiquetas))
    return {"nombre": str(nombres[indice]) if proporcion >= .45 else "indeterminado",
            "proporcion": round(proporcion, 3)}
