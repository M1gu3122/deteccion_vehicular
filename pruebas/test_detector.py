"""Prueba el detector best.pt sobre imagenes de auto real y de repuesto."""
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
modelo = YOLO(str(BASE / "modelos" / "best.pt"))
print("Clases del detector:", modelo.names)

pruebas = [
    ("AUTO REAL (sorento calle)", BASE / "dataset_autos" / "kia_sorento" / "sorento_calle_00030.jpg"),
    ("REPUESTO (amanti 2004)", BASE / "dataset_autos" / "amanti" / "amanti_amanti_2004_00106.jpg"),
    ("AUTO (picanto studio)", BASE / "dataset_autos" / "train" / "kia_picanto" / "kia_picanto_0299d530f3e3c0be.jpg"),
]

for etiqueta, ruta in pruebas:
    if not ruta.exists():
        print(f"\n[{etiqueta}] NO EXISTE: {ruta.name}")
        continue
    res = modelo.predict(str(ruta), conf=0.25, verbose=False)[0]
    print(f"\n[{etiqueta}] {ruta.name}")
    if res.boxes is not None and len(res.boxes) > 0:
        for i, caja in enumerate(res.boxes):
            clase = modelo.names[int(caja.cls)]
            conf = float(caja.conf)
            x1, y1, x2, y2 = caja.xyxy[0].tolist()
            print(f"   det#{i}: {clase} conf={conf:.2f} area_rel={((x2-x1)*(y2-y1))/max(1,(res.orig_shape[0]*res.orig_shape[1])):.3f}")
    else:
        print("   SIN DETECCION (imagen descartable)")
