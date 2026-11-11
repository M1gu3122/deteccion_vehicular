"""Lista las fuentes/dominios y consultas por clase para construir el filtro."""
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos")

CLASES = {
    'amanti': 'amanti', 'forte': 'forte', 'optima': 'optima', 'picanto': 'picanto',
    'rio': 'rio', 'sedona': 'sedona', 'sephia': 'sephia', 'sorento': 'kia_sorento',
    'soul': 'soul', 'spectra': 'kia_spectra', 'sportage': 'sportage',
}

dominios = Counter()
consultas = Counter()
fuentes = Counter()

for clase, carpeta in CLASES.items():
    mj = BASE / carpeta / 'metadata.jsonl'
    if not mj.exists():
        continue
    for linea in mj.open(encoding='utf-8'):
        try:
            r = json.loads(linea)
        except json.JSONDecodeError:
            continue
        url = r.get('url_origen') or r.get('url_pagina') or r.get('url_imagen') or ''
        dom = urlparse(url).netloc.lower().replace('www.', '')
        dominios[dom] += 1
        consultas[(clase, str(r.get('consulta', '')))] += 1
        fuentes[str(r.get('fuente', ''))] += 1

print("=" * 80)
print("DOMINIOS (top 45)")
print("=" * 80)
for dom, n in dominios.most_common(45):
    print(f"  {n:>5}  {dom}")

print("\n" + "=" * 80)
print("CONSULTAS QUE PRODUCEN CONTENIDO IRRELEVANTE")
print("=" * 80)
for (clase, cons), n in sorted(consultas.items()):
    c = cons.lower()
    if any(k in c for k in ['repuesto', 'part', 'interior', 'tablero', 'asiento',
                            'motor', 'accesorio', 'filtro', 'freno', 'pastilla',
                            'buje', 'faro', 'llanta', 'rin', 'bateria', 'sensor',
                            'precio', 'venta', 'oferta', 'ficha', 'tecnico']):
        print(f"  {clase:<12} {cons:<38} {n}")

print("\n" + "=" * 80)
print("FUENTES (top 30)")
print("=" * 80)
for f, n in fuentes.most_common(30):
    print(f"  {n:>5}  {f[:70]}")
