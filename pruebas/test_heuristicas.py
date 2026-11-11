"""
Prueba heuristicas para separar interiores de exteriores en recortes de vehiculo.

Hipotesis:
  - El detector YOLO marca casi todo el frame cuando la foto es un interior
    (la camara esta dentro del auto, asi que el "vehiculo" ocupa todo).
  - Los exteriores tienen cielo (azul claro), vegetacion (verde) o asfalto.

Salida: features de cada imagen para elegir umbrales.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
from preparar_dataset_v2 import (CARPETAS_ORIGEN, cargar_metadatos,
                                 hash_md5, razon_rechazo)

BASE = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
ORIGEN = BASE / "dataset_autos"
MODELO = YOLO(str(BASE / "modelos" / "best.pt"))

# La imagen interior confirmada (reconstruida) y su origen
INTERIOR_CONOCIDO = BASE / "dataset_autos_v2" / "val" / "kia_amanti" / "kia_amanti_0002.jpg"


def features(imagen, caja):
    """Devuelve (ocupacion_caja, frac_cielo, frac_vegetacion, saturacion_media)."""
    alto, ancho = imagen.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in caja]
    ocupacion = max(0.0, (x2 - x1) * (y2 - y1) / max(1, ancho * alto))

    recorte = imagen[max(0, y1):y2, max(0, x1):x2]
    if recorte.size == 0:
        return ocupacion, 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Cielo: azul (H 100-130 en OpenCV) con brillo alto
    cielo = ((h >= 100) & (h <= 135) & (s >= 40) & (v >= 140)).sum()
    frac_cielo = cielo / recorte.shape[0] / recorte.shape[1]

    # Vegetacion: verde (H 35-85) con saturacion media-alta
    veg = ((h >= 35) & (h <= 85) & (s >= 50) & (v >= 60)).sum()
    frac_veg = veg / recorte.shape[0] / recorte.shape[1]

    sat_media = float(s.mean()) / 255.0
    return ocupacion, frac_cielo, frac_veg, sat_media


def analizar(ruta):
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        return None
    res = MODELO.predict(imagen, conf=0.50, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    mejor = max(res.boxes, key=lambda b: float(b.cls) * 0 + (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
    caja = mejor.xyxy[0].tolist()
    conf = float(mejor.conf)
    return features(imagen, caja) + (conf,)


print("=" * 92)
print("IMAGEN INTERIOR CONFIRMADO (cropped)")
print("=" * 92)
r = analizar(INTERIOR_CONOCIDO)
if r:
    print(f"  ocupacion_caja={r[0]:.3f}  frac_cielo={r[1]:.3f}  "
          f"frac_veg={r[2]:.3f}  sat_media={r[3]:.3f}  conf={r[4]:.2f}")

print("\n" + "=" * 92)
print("MUESTRA DE EXTERIORES (pool original filtrado)")
print("=" * 92)
print(f"{'clase':<10} {'archivo':<34} {'ocup':>6} {'cielo':>7} {'veg':>7} {'sat':>6}")
print("-" * 92)

muestras = []
for carpeta_nombre, clase in CARPETAS_ORIGEN.items():
    carpeta = ORIGEN / carpeta_nombre
    if not carpeta.is_dir():
        continue
    metadatos = cargar_metadatos(carpeta)
    archivos = [p for p in carpeta.iterdir()
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    validos = [p for p in sorted(archivos)
               if not razon_rechazo(metadatos.get(p.name, {}), clase)]
    # tomar 6 por clase, espaciados
    paso = max(1, len(validos) // 6)
    for ruta in validos[::paso][:6]:
        f = analizar(ruta)
        if not f:
            continue
        muestras.append((clase, ruta.name, f))
        print(f"{clase:<10} {ruta.name[:33]:<34} {f[0]:>6.3f} {f[1]:>7.3f} "
              f"{f[2]:>7.3f} {f[3]:>6.3f}")

print("\n" + "=" * 92)
print("RESUMEN DE OCUPACION DE CAJA")
print("=" * 92)
ocups = sorted(f[2][0] for f in muestras)
if ocups:
    print(f"  min={ocups[0]:.3f}  p25={ocups[len(ocups)//4]:.3f}  "
          f"mediana={ocups[len(ocups)//2]:.3f}  p75={ocups[3*len(ocups)//4]:.3f}  "
          f"max={ocups[-1]:.3f}")
    print(f"  >0.85: {sum(1 for o in ocups if o > 0.85)} de {len(ocups)}")
