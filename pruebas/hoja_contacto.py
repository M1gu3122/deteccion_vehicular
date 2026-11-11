"""Genera hojas de contacto para inspeccionar visualmente el dataset v2."""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")


def hoja_contacto(origen, destino, clases, por_clase=6, lado=200, seed=7):
    random.seed(seed)
    celdas = []
    etiquetas = []

    for clase in clases:
        carpeta = origen / "val" / f"kia_{clase}"
        if not carpeta.is_dir():
            continue
        archivos = sorted(carpeta.glob("*.jpg"))
        if not archivos:
            continue
        eleccion = random.sample(archivos, min(por_clase, len(archivos)))
        for ruta in eleccion:
            img = cv2.imread(str(ruta))
            if img is None:
                continue
            img = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_AREA)
            celdas.append(img)
            etiquetas.append(clase)

    if not celdas:
        print("No hay imagenes")
        return

    columnas = 6
    filas = (len(celdas) + columnas - 1) // columnas
    alto_etiqueta = 24

    lienzo = np.full((filas * (lado + alto_etiqueta), columnas * lado, 3),
                     255, dtype=np.uint8)

    for i, (img, etiqueta) in enumerate(zip(celdas, etiquetas)):
        f, c = divmod(i, columnas)
        y = f * (lado + alto_etiqueta)
        x = c * lado
        lienzo[y:y + lado, x:x + lado] = img
        cv2.putText(lienzo, etiqueta.replace('kia_', ''), (x + 5, y + lado + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(destino), lienzo)
    print(f"Hoja guardada: {destino}  ({len(celdas)} imagenes, {filas}x{columnas})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origen", default=str(BASE / "dataset_autos_v2"))
    parser.add_argument("--salida", default=str(BASE / "hoja_contacto.jpg"))
    parser.add_argument("--por-clase", type=int, default=6)
    parser.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args()

    CLASES = ['amanti', 'forte', 'optima', 'picanto', 'rio', 'sedona',
              'sephia', 'sorento', 'soul', 'spectra', 'sportage']

    hoja_contacto(Path(args.origen), Path(args.salida), CLASES,
                  por_clase=args.por_clase, seed=args.semilla)
