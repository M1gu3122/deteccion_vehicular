"""Analiza metadata.jsonl para clasificar imagenes: vehiculo real vs repuestos/irrelevantes."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
DATASET_DIR = BASE_DIR / "dataset_autos"

CLASES = ['amanti', 'forte', 'optima', 'picanto', 'rio', 'sedona',
          'sephia', 'sorento', 'soul', 'spectra', 'sportage']

# Palabras que indican que NO es una foto de un vehiculo completo
PALABRAS_REPUESTO = [
    'repuesto', 'repuestos', 'part', 'parts', 'accesorio', 'accesorios',
    'cambio', 'cambios', 'filtro', 'filtros', 'buje', 'bujes', 'bamper',
    'bumper', 'capot', 'bonete', 'faro', 'faros', 'foco', 'focus', 'pila',
    'bateria', 'baterias', 'motor', 'caja', 'transmision', 'embrague',
    'amortiguador', 'amortiguadores', 'espiral', 'bujia', 'bujias',
    'correa', 'pastilla', 'pastillas', 'disco', 'freno', 'frenos',
    'radiador', 'condensador', 'compresor', 'alternador', 'arranque',
    'manguera', 'mangueras', 'filtro', 'suspension', 'direccion',
    'espejo', 'espejos', 'farola', 'pilas', 'sensor', 'sensores',
    'kit', 'juego', 'set', 'original', 'genuine', 'oem', 'replacement',
    'catálogo', 'catalogo', 'ficha', 'manual', 'diagrama', 'plano',
    'rueda', 'llanta', 'llantas', 'rin', 'rines', 'neumatico', 'neumaticos',
    'moldura', 'embellecedor', 'protector', 'alfombra', 'tapiceria',
    'asiento', 'asientos', 'motor', 'cabezal', 'tablero', 'accesorio',
    'wholesale', 'supplier', 'alibaba', 'mercadolibre', 'aliexpress',
]

# Palabras que SI son vehiculo real
PALABRAS_VEHICULO = [
    'kia', 'picanto', 'sorento', 'sportage', 'sedona', 'soul', 'rio',
    'optima', 'sephia', 'spectra', 'amanti', 'forte', 'carro', 'vehículo',
    'vehiculo', 'automovil', 'automóvil', 'suv', 'hatchback', 'sedan',
]


def clasificar_registro(registro):
    titulo = str(registro.get('titulo', '')).lower()
    fuente = str(registro.get('fuente', '')).lower()
    etiqueta = str(registro.get('etiqueta', '')).lower()
    enlace = str(registro.get('url_pagina', '')).lower()
    texto = f"{titulo} {fuente} {etiqueta} {enlace}"

    for palabra in PALABRAS_REPUESTO:
        if palabra in texto:
            return 'REPUESTO', palabra

    if any(p in texto for p in PALABRAS_VEHICULO):
        return 'VEHICULO', ''

    return 'AMBIGUO', ''


print("=" * 82)
print("ANALISIS DE METADATOS: VEHICULO REAL vs REPUESTO/IRRELEVANTE")
print("=" * 82)

resumen_global = Counter()
inventario = {}

for clase in CLASES:
    # Buscar la carpeta raiz que contenga metadata.jsonl
    candidatas = [DATASET_DIR / clase, DATASET_DIR / f"kia_{clase}"]
    carpeta = next((c for c in candidatas if (c / 'metadata.jsonl').exists()), None)

    if carpeta is None:
        print(f"\n{clase}: SIN metadata.jsonl")
        continue

    conteo = Counter()
    ejemplos = defaultdict(list)
    con_meta = 0

    for linea in (carpeta / 'metadata.jsonl').open(encoding='utf-8'):
        try:
            registro = json.loads(linea)
        except json.JSONDecodeError:
            continue
        con_meta += 1
        tipo, palabra = clasificar_registro(registro)
        conteo[tipo] += 1
        if len(ejemplos[tipo]) < 5:
            ejemplos[tipo].append((registro.get('archivo', ''),
                                   str(registro.get('titulo', ''))[:70]))

    # Contar archivos reales en disco
    archivos_en_disco = [p for p in carpeta.iterdir()
                         if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]

    inventario[clase] = {
        'disco': len(archivos_en_disco),
        'meta': con_meta,
        'vehiculo': conteo['VEHICULO'],
        'repuesto': conteo['REPUESTO'],
        'ambiguo': conteo['AMBIGUO'],
        'carpeta': carpeta.name,
    }
    resumen_global.update(conteo)

    print(f"\n[{clase}]  carpeta='{carpeta.name}'  archivos={len(archivos_en_disco)}  metadata={con_meta}")
    print(f"    VEHICULO={conteo['VEHICULO']:<5} REPUESTO={conteo['REPUESTO']:<5} AMBIGUO={conteo['AMBIGUO']}")
    for tipo in ('REPUESTO', 'AMBIGUO'):
        for arch, tit in ejemplos[tipo][:3]:
            print(f"      [{tipo}] {arch[:34]:<34} {tit}")

print("\n" + "=" * 82)
print("TOTAL GLOBAL")
print("=" * 82)
print(f"  VEHICULO:  {resumen_global['VEHICULO']}")
print(f"  REPUESTO:  {resumen_global['REPUESTO']}")
print(f"  AMBIGUO:   {resumen_global['AMBIGUO']}")
total = sum(resumen_global.values())
if total:
    pct = resumen_global['REPUESTO'] / total * 100
    print(f"  Contaminacion por repuestos: {pct:.1f}%")
