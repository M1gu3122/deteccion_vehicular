"""Diagnostico completo del dataset de clasificacion Kia."""
import hashlib
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

BASE_DIR = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
DATASET_DIR = BASE_DIR / "dataset_autos"

CLASES = ['amanti', 'forte', 'optima', 'picanto', 'rio', 'sedona',
          'sephia', 'sorento', 'soul', 'spectra', 'sportage']

EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def hash_archivo(ruta):
    h = hashlib.md5()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(65536), b''):
            h.update(bloque)
    return h.hexdigest()


def listar(carpeta):
    if not carpeta.is_dir():
        return []
    return sorted([p for p in carpeta.iterdir() if p.suffix.lower() in EXT])


print("=" * 78)
print("DIAGNOSTICO DEL DATASET")
print("=" * 78)

# ---------------------------------------------------------------- 1. INVENTARIO
print("\n[1] INVENTARIO POR ORIGEN")
print("-" * 78)
print(f"{'clase':<12} {'raiz':>6} {'train':>6} {'val':>6} {'total':>7}")
print("-" * 78)

pool_por_clase = {}
for clase in CLASES:
    raiz = listar(DATASET_DIR / clase)
    raiz_kia = listar(DATASET_DIR / f"kia_{clase}")
    train = listar(DATASET_DIR / "train" / f"kia_{clase}")
    val = listar(DATASET_DIR / "val" / f"kia_{clase}")
    total = len(raiz) + len(raiz_kia) + len(train) + len(val)
    pool_por_clase[clase] = raiz + raiz_kia + train + val
    print(f"{clase:<12} {len(raiz):>6} {len(raiz_kia):>6} {len(train):>6} {len(val):>6} {total:>7}")

# ----------------------------------------------- 2. DUPLICADOS POR CONTENIDO
print("\n[2] DUPLICADOS POR CONTENIDO (mismo hash MD5)")
print("-" * 78)

hash_global = defaultdict(list)   # hash -> [(clase, ruta)]
hash_por_clase = {}
for clase in CLASES:
    por_clase = defaultdict(list)
    for ruta in pool_por_clase[clase]:
        h = hash_archivo(ruta)
        hash_global[h].append((clase, ruta))
        por_clase[h].append(ruta)
    hash_por_clase[clase] = por_clase

duplicados_totales = 0
duplicados_entre_clases = []
for h, lista in hash_global.items():
    if len(lista) > 1:
        duplicados_totales += len(lista) - 1
        clases_involucradas = {c for c, _ in lista}
        if len(clases_involucradas) > 1:
            duplicados_entre_clases.append((h, lista))

print(f"Imagenes duplicadas (copias repetidas): {duplicados_totales}")
print(f"Grupos con la MISMA imagen en CLASES DIFERENTES: {len(duplicados_entre_clases)}")
for h, lista in duplicados_entre_clases[:10]:
    print(f"  {h[:10]}:")
    for clase, ruta in lista:
        print(f"      {clase:<12} {ruta.name}")

print("\nDuplicados DENTRO de cada clase:")
for clase in CLASES:
    dups = sum(len(v) - 1 for v in hash_por_clase[clase].values() if len(v) > 1)
    unicos = len(hash_por_clase[clase])
    flag = "  <-- ALERTA" if dups > 0 else ""
    print(f"  {clase:<12} unicas={unicos:<5} repetidas={dups}{flag}")

# ------------------------------------------- 3. FUGA ENTRE TRAIN Y VAL
print("\n[3] FUGA DE DATOS (misma imagen en train y val)")
print("-" * 78)
fugas = 0
for clase in CLASES:
    hashes_train = set()
    for ruta in listar(DATASET_DIR / "train" / f"kia_{clase}"):
        hashes_train.add(hash_archivo(ruta))
    hashes_val = set()
    for ruta in listar(DATASET_DIR / "val" / f"kia_{clase}"):
        hashes_val.add(hash_archivo(ruta))
    comunes = hashes_train & hashes_val
    if comunes:
        fugas += len(comunes)
        print(f"  {clase:<12} {len(comunes)} imagenes en train Y val")
if fugas == 0:
    print("  Sin coincidencias exactas entre train y val")
else:
    print(f"\n  TOTAL fuga exacta: {fugas} imagenes")

# ------------------------------------------- 4. INTEGRIDAD DE ARCHIVOS
print("\n[4] INTEGRIDAD DE ARCHIVOS")
print("-" * 78)
corruptos = []
dimensiones = defaultdict(list)
for clase in CLASES:
    for ruta in pool_por_clase[clase]:
        try:
            with Image.open(ruta) as img:
                img.verify()
            with Image.open(ruta) as img:
                dimensiones[clase].append(img.size)
        except Exception as e:
            corruptos.append((clase, ruta.name, str(e)))

if corruptos:
    print(f"Archivos CORRUPTOS: {len(corruptos)}")
    for clase, nombre, err in corruptos[:15]:
        print(f"  {clase}/{nombre}: {err}")
else:
    print("Todos los archivos son imagenes validas")

print("\nTamanos (ancho x alto) por clase:")
for clase in CLASES:
    dims = dimensiones[clase]
    if not dims:
        continue
    anchos = [d[0] for d in dims]
    altos = [d[1] for d in dims]
    unicas = len(set(dims))
    print(f"  {clase:<12} n={len(dims):<4} ancho {min(anchos):>4}-{max(anchos):<5} "
          f"alto {min(altos):>4}-{max(altos):<5} formatos_distintos={unicas}")

# ------------------------------------------- 5. CANTIDAD DE VAL
print("\n[5] EVALUACION DE CONJUNTO DE VALIDACION")
print("-" * 78)
print(f"{'clase':<12} {'train':>6} {'val':>6} {'val%':>7} {'imgs/clase val':>15}")
print("-" * 78)
for clase in CLASES:
    n_train = len(listar(DATASET_DIR / "train" / f"kia_{clase}"))
    n_val = len(listar(DATASET_DIR / "val" / f"kia_{clase}"))
    total = n_train + n_val
    pct = (n_val / total * 100) if total else 0
    print(f"{clase:<12} {n_train:>6} {n_val:>6} {pct:>6.1f}% {n_val:>15}")

print("\nResumen:")
print(f"  Total train: {sum(len(listar(DATASET_DIR/'train'/f'kia_{c}')) for c in CLASES)}")
print(f"  Total val:   {sum(len(listar(DATASET_DIR/'val'/f'kia_{c}')) for c in CLASES)}")
print("=" * 78)
