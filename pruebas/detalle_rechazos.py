"""Detalla los motivos de rechazo por clase para refinar los filtros."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
from preparar_dataset_v2 import (CARPETAS_ORIGEN, CARPETAS_ORIGEN as CO,
                                 cargar_metadatos, razon_rechazo)

BASE = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos")

for carpeta_nombre, clase in CARPETAS_ORIGEN.items():
    carpeta = BASE / carpeta_nombre
    if not carpeta.is_dir():
        continue
    metadatos = cargar_metadatos(carpeta)
    archivos = [p for p in carpeta.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    razones = Counter()
    consultas = Counter()
    for ruta in archivos:
        reg = metadatos.get(ruta.name, {})
        motivo = razon_rechazo(reg)
        if motivo:
            razones[motivo] += 1
        consultas[str(reg.get('consulta', ''))] += 1

    print(f"\n{'='*70}\n{clase}  (carpeta={carpeta_nombre}, total={len(archivos)})")
    print(f"{'='*70}")
    print("Razones de rechazo:")
    for r, n in razones.most_common(10):
        print(f"   {n:>4}  {r}")
    print("Consultas (top 12):")
    for c, n in consultas.most_common(12):
        print(f"   {n:>4}  '{c}'")
