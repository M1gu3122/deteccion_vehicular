"""
Reconstruye el dataset de clasificacion Kia desde cero.

Problema que corrige:
  El dataset original (dataset_autos/train + val) tenia solo 255 imagenes de
  entrenamiento con recortes de calidad desigual (renders limpios de Picanto
  frente a recortes de 100x100 px de otras clases). El modelo aprendio "imagen
  limpia y grande = Picanto" y por eso devolvia Picanto con 99.9% de confianza
  para cualquier entrada.

Que hace este script:
  1. Reúne el pool completo (~2796 imagenes) de las carpetas raiz descargadas.
  2. Elimina duplicados exactos (MD5), incluidos los repetidos en dos clases.
  3. Descarta imagenes de repuestos, interiores y catalogs usando los metadatos
     de descarga (dominio de la fuente y texto de la consulta).
  4. Verifica con el detector YOLO (best.pt) que la imagen contenga un vehiculo
     y recorta el auto con margen.
  5. Normaliza todos los recortes a 224x224 para que ninguna clase domine por
     calidad de imagen.
  6. Divide 80/20 por clase (estratificado) y escribe data.yaml.

Uso:
    python preparar_dataset_v2.py
    python preparar_dataset_v2.py --frac-val 0.25 --min-conf 0.45
    python preparar_dataset_v2.py --sin-detector
"""

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
ORIGEN_DIR = BASE_DIR / "dataset_autos"
DESTINO_DIR = BASE_DIR / "dataset_autos_v2"
MODELO_DETECTOR = BASE_DIR / "modelos" / "best.pt"

# Carpeta raiz de descarga -> clase del modelo
CARPETAS_ORIGEN = {
    'amanti': 'amanti',
    'forte': 'forte',
    'optima': 'optima',
    'picanto': 'picanto',
    'rio': 'rio',
    'sedona': 'sedona',
    'sephia': 'sephia',
    'kia_sorento': 'sorento',
    'soul': 'soul',
    'kia_spectra': 'spectra',
    'sportage': 'sportage',
}

CLASES = ['amanti', 'forte', 'optima', 'picanto', 'rio', 'sedona',
          'sephia', 'sorento', 'soul', 'spectra', 'sportage']

EXTENSIONES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Dominios que venden repuestos, stock o renders, no vehiculos reales
DOMINIOS_BLOQUEADOS = {
    'ebay.com', 'co.ebay.com', 'amazon.com', 'walmart.com', 'go-parts.com',
    'alamy.com', 'dreamstime.com', 'es.dreamstime.com', '3dmodels.org',
    'automotive24.center', 'carpro.com', 'partsavatar.com', 'megazip.net',
    'partshub.store', 'autoparts.com', 'rockauto.com', 'jegs.com',
    'summitracing.com', 'carid.com', 'autozone.com', 'oreillyauto.com',
    'pepboys.com', 'advanceautoparts.com', 'mistercarparts.com',
    'mercadolibre.com', 'mercadolibre.com.co', 'aliexpress.com',
    'alibaba.com', 'buscalibre.com', 'lincolnparts.com', 'checkersmall.com',
    'napaonline.com', 'oreilly.com', 'amayama.com', 'fitsera.com',
    'tradek.com', 'vividracing.com', 'olx.com', 'tucarro.com',
}

# Palabras en la consulta o el titulo que indican que no es un auto exterior
# Las frases (con espacio) se buscan como subcadena.
# Las palabras simples se buscan con limites de palabra para no capturar
# falsos positivos como "MotorTrend" o "participantes".
PALABRAS_IRTECNANTES = [
    # interiores
    'interior', 'tablero', 'asiento', 'asientos', 'segunda fila',
    'tercera fila', 'consola central', 'salpicadero', 'cajuela', 'maletero',
    'volante', 'radio', 'pantalla', 'reproductor', 'airbag',
    # repuestos y mecanica
    'repuesto', 'repuestos', 'accesorio', 'accesorios', 'despiece',
    'recambio', 'refaccion', 'refacciones', 'exploded view',
    'vista explotada', 'motor de', 'motor kia', 'caja de cambios',
    'transmision', 'embrague', 'bamper', 'bumper', 'capot', 'bonete',
    'faro', 'faros', 'foco', 'llanta', 'llantas', 'rin', 'rines',
    'neumatico', 'neumaticos', 'moldura', 'alfombra', 'tapiceria',
    'amortiguador', 'amortiguadores', 'espiral', 'bujia', 'bujias',
    'correa', 'pastilla', 'pastillas', 'freno', 'frenos', 'radiador',
    'condensador', 'compresor', 'alternador', 'arranque', 'manguera',
    'mangueras', 'suspension', 'direccion', 'espejo', 'espejos', 'farola',
    'piloto', 'bateria', 'baterias', 'sensor', 'sensores', 'buje', 'bujes',
    'filtro', 'filtros', 'ventilador', 'bomba', 'valvula', 'junta',
    'reten', 'rodamiento', 'intercooler', 'radiador',
    # referencias comerciales o documental
    'manual', 'ficha tecnica', 'datos tecnicos', 'precio', 'cotiza',
    'financiamiento', 'catalogo', 'diagrama', 'plano', 'aseguradora',
]

# Palabras que SI son validas para un auto exterior
PALABRAS_VEHICULO = [
    'frontal', 'lateral', 'trasera', 'tres cuartos', 'calle', 'carro',
    'vehiculo', 'automovil', 'suv', 'hatchback', 'sedan', 'kia',
]

# Consultas que no corresponden al modelo de su carpeta (etiqueta contaminada)
MODELOS_FUERA_DE_CLASE = {
    'amanti': ['opirus', 'optima', 'rio', 'sorento', 'sportage', 'sedona',
               'soul', 'picanto', 'spectra', 'sephia', 'forte'],
    'forte': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
              'picanto', 'spectra', 'sephia', 'rio'],
    'optima': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
               'picanto', 'spectra', 'sephia', 'forte'],
    'picanto': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
                'spectra', 'sephia', 'forte', 'rio'],
    'rio': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
            'picanto', 'spectra', 'sephia', 'forte'],
    'sedona': ['opirus', 'amanti', 'sorento', 'sportage', 'soul', 'picanto',
               'spectra', 'sephia', 'forte', 'rio'],
    'sephia': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
               'picanto', 'spectra', 'forte', 'rio', 'shuma'],
    'sorento': ['opirus', 'amanti', 'sedona', 'sportage', 'soul', 'picanto',
                'spectra', 'sephia', 'forte', 'rio'],
    'soul': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'picanto',
             'spectra', 'sephia', 'forte', 'rio'],
    'spectra': ['opirus', 'amanti', 'sedona', 'sorento', 'sportage', 'soul',
                'picanto', 'sephia', 'forte', 'rio'],
    'sportage': ['opirus', 'amanti', 'sedona', 'sorento', 'soul', 'picanto',
                 'spectra', 'sephia', 'forte', 'rio'],
}

# Consultas ambiguas: la imagen puede mostrar varios vehiculos
CONSULTAS_AMBIGUAS = [
    'varios_vehiculos', 'varios autos', 'multimarca', 'concesionario autos',
    'autolote', 'parking', 'parqueadero autos', 'galeria', 'multimarca',
]


def hash_md5(ruta):
    h = hashlib.md5()
    with open(ruta, 'rb') as f:
        for bloque in iter(lambda: f.read(65536), b''):
            h.update(bloque)
    return h.hexdigest()


def cargar_metadatos(carpeta):
    """Devuelve {nombre_archivo: registro} leyendo metadata.jsonl."""
    ruta_meta = carpeta / 'metadata.jsonl'
    registros = {}
    if not ruta_meta.exists():
        return registros
    with ruta_meta.open(encoding='utf-8') as f:
        for linea in f:
            try:
                reg = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if isinstance(reg, dict) and reg.get('archivo'):
                registros[reg['archivo']] = reg
    return registros


def razon_rechazo(registro, clase):
    """Devuelve el motivo del rechazo o None si la imagen es valida."""
    if not registro:
        return None

    url = (registro.get('url_origen') or registro.get('url_pagina')
           or registro.get('url_imagen') or '')
    dominio = urlparse(url).netloc.lower().replace('www.', '')
    for bloqueado in DOMINIOS_BLOQUEADOS:
        if dominio == bloqueado or dominio.endswith('.' + bloqueado):
            return f'dominio:{dominio}'

    consulta = str(registro.get('consulta', '')).lower()
    titulo = str(registro.get('titulo', '')).lower()
    etiqueta = str(registro.get('etiqueta', '')).lower()
    texto = f"{consulta} {titulo} {etiqueta}"

    for palabra in PALABRAS_IRTECNANTES:
        if ' ' in palabra:
            coincide = palabra in texto
        else:
            coincide = re.search(r'\b' + re.escape(palabra) + r'\b', texto) is not None
        if coincide:
            return f'texto:{palabra}'

    # Consultas ambiguas: pueden mostrar varios vehiculos de modelos distintos
    for palabra in CONSULTAS_AMBIGUAS:
        if palabra in consulta or palabra in titulo:
            return f'ambigua:{palabra}'

    # La consulta busco otro modelo Kia distinto al de la carpeta
    for otro in MODELOS_FUERA_DE_CLASE.get(clase, []):
        if re.search(r'\b' + re.escape(otro) + r'\b', consulta) or \
           re.search(r'\b' + re.escape(otro) + r'\b', etiqueta):
            return f'modelo-ajeno:{otro}'

    # La regla "sin-vehiculo" solo aplica al pipeline Serper (tiene titulo).
    # En el pipeline por modelo la consulta siempre es un termino valido
    # ("lluvia", "calle", "frontal"), asi que no se puede exigir la palabra.
    if registro.get('titulo') is not None or registro.get('etiqueta') is not None:
        hay_vehiculo = any(p in texto for p in PALABRAS_VEHICULO)
        if texto.strip() and not hay_vehiculo:
            return 'texto:sin-vehiculo'

    return None


def recortar_vehiculo(modelo, imagen, min_conf, margen, lado_minimo):
    """Detecta el vehiculo y devuelve el recorte, o None."""
    alto, ancho = imagen.shape[:2]
    resultado = modelo.predict(imagen, conf=min_conf, verbose=False)[0]

    if resultado.boxes is None or len(resultado.boxes) == 0:
        return None, 'sin-deteccion'

    mejor = None
    mejor_area = -1
    for caja in resultado.boxes:
        conf = float(caja.conf)
        x1, y1, x2, y2 = caja.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > mejor_area:
            mejor_area = area
            mejor = (x1, y1, x2, y2, conf)

    if mejor is None:
        return None, 'sin-deteccion'

    x1, y1, x2, y2, conf = mejor

    # La caja debe cubrir una parte significativa de la imagen
    proporcion = mejor_area / max(1, ancho * alto)
    if proporcion < 0.08:
        return None, 'vehiculo-pequeno'

    mx = int((x2 - x1) * margen)
    my = int((y2 - y1) * margen)
    x1 = max(0, int(x1) - mx)
    y1 = max(0, int(y1) - my)
    x2 = min(ancho, int(x2) + mx)
    y2 = min(alto, int(y2) + my)

    recorte = imagen[y1:y2, x1:x2]
    if recorte.size == 0:
        return None, 'recorte-vacio'

    alto_r, ancho_r = recorte.shape[:2]
    if min(alto_r, ancho_r) < lado_minimo:
        return None, f'recorte-pequeno:{min(alto_r, ancho_r)}'

    return recorte, None


def normalizar(imagen, lado):
    """Redimensiona manteniendo proporcion y rellena con color de borde."""
    alto, ancho = imagen.shape[:2]
    escala = lado / max(alto, ancho)
    nuevo_alto = max(1, int(round(alto * escala)))
    nuevo_ancho = max(1, int(round(ancho * escala)))
    redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto),
                                interpolation=cv2.INTER_AREA)
    lienzo = np.full((lado, lado, 3), 114, dtype=np.uint8)
    y = (lado - nuevo_alto) // 2
    x = (lado - nuevo_ancho) // 2
    lienzo[y:y + nuevo_alto, x:x + nuevo_ancho] = redimensionada
    return lienzo


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruye el dataset de clasificacion Kia con filtros de calidad."
    )
    parser.add_argument("--origen", default=str(ORIGEN_DIR))
    parser.add_argument("--destino", default=str(DESTINO_DIR))
    parser.add_argument("--frac-val", type=float, default=0.20)
    parser.add_argument("--min-conf", type=float, default=0.50)
    parser.add_argument("--margen", type=float, default=0.08)
    parser.add_argument("--lado", type=int, default=224)
    parser.add_argument("--lado-minimo", type=int, default=120)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-detector", action="store_true",
                        help="No recorta ni verifica con el detector YOLO.")
    args = parser.parse_args()

    origen = Path(args.origen)
    destino = Path(args.destino)

    if not origen.is_dir():
        print(f"No existe la carpeta de origen: {origen}")
        return

    modelo = None
    if not args.sin_detector:
        if not MODELO_DETECTOR.exists():
            print(f"No se encontro el detector: {MODELO_DETECTOR}")
            print("Ejecuta con --sin-detector para omitir el recorte.")
            return
        modelo = YOLO(str(MODELO_DETECTOR))
        print(f"Detector cargado: {MODELO_DETECTOR.name}")

    random.seed(args.semilla)

    # --------------------------------------------------- recolectar el pool
    print("\n" + "=" * 74)
    print("PASO 1: RECOLECTAR POOL COMPLETO")
    print("=" * 74)

    pool = defaultdict(list)      # clase -> [(ruta, registro)]
    hashes_vistos = {}            # md5 -> (clase, ruta)
    duplicados = 0
    conflictos = 0

    for carpeta_nombre, clase in CARPETAS_ORIGEN.items():
        carpeta = origen / carpeta_nombre
        if not carpeta.is_dir():
            print(f"  [AVISO] No existe: {carpeta}")
            continue

        metadatos = cargar_metadatos(carpeta)
        archivos = [p for p in carpeta.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTENSIONES]

        for ruta in sorted(archivos):
            md5 = hash_md5(ruta)
            if md5 in hashes_vistos:
                duplicados += 1
                clase_previa, _ = hashes_vistos[md5]
                if clase_previa != clase:
                    conflictos += 1
                continue
            hashes_vistos[md5] = (clase, ruta)
            pool[clase].append((ruta, metadatos.get(ruta.name, {})))

    total_pool = sum(len(v) for v in pool.values())
    print(f"  Imagenes únicas en el pool: {total_pool}")
    print(f"  Duplicados descartados: {duplicados}")
    print(f"  Conflictos entre clases: {conflictos}")

    # --------------------------------------------------- filtrar y recortar
    print("\n" + "=" * 74)
    print("PASO 2: FILTRAR POR METADATOS")
    print("=" * 74)

    pool_filtrado = defaultdict(list)
    rechazos = defaultdict(int)
    for clase, items in pool.items():
        for ruta, registro in items:
            motivo = razon_rechazo(registro, clase)
            if motivo:
                rechazos[motivo.split(':')[0]] += 1
                continue
            pool_filtrado[clase].append((ruta, registro))
        antes = len(items)
        despues = len(pool_filtrado[clase])
        print(f"  {clase:<12} {antes:>4} -> {despues:>4}")

    print("\n  Motivos de rechazo:")
    for motivo, n in sorted(rechazos.items(), key=lambda x: -x[1]):
        print(f"    {motivo:<16} {n}")

    # --------------------------------------------------- recortar y normalizar
    print("\n" + "=" * 74)
    print("PASO 3: RECORTAR Y NORMALIZAR")
    print("=" * 74)

    if destino.exists():
        shutil.rmtree(destino)
    for clase in CLASES:
        (destino / "train" / f"kia_{clase}").mkdir(parents=True, exist_ok=True)
        (destino / "val" / f"kia_{clase}").mkdir(parents=True, exist_ok=True)

    guardados = defaultdict(list)
    fallidos = defaultdict(int)

    for clase in CLASES:
        items = pool_filtrado.get(clase, [])
        n_ok = 0
        for indice, (ruta, _) in enumerate(items, 1):
            imagen = cv2.imread(str(ruta))
            if imagen is None:
                fallidos['ilegible'] += 1
                continue

            if modelo is not None:
                recorte, motivo = recortar_vehiculo(
                    modelo, imagen, args.min_conf, args.margen, args.lado_minimo)
                if recorte is None:
                    fallidos[motivo.split(':')[0]] += 1
                    continue
            else:
                recorte = imagen

            recortada = normalizar(recorte, args.lado)
            nombre = f"kia_{clase}_{indice:04d}.jpg"
            cv2.imwrite(str(destino / nombre), recortada)
            guardados[clase].append(nombre)
            n_ok += 1

            if indice % 50 == 0:
                print(f"    {clase}: {n_ok}/{len(items)}")

        print(f"  {clase:<12} procesadas={len(items):<5} guardadas={n_ok:<5} "
              f"descartadas={len(items) - n_ok}")

    print("\n  Motivos de descarte por deteccion:")
    for motivo, n in sorted(fallidos.items(), key=lambda x: -x[1]):
        print(f"    {motivo:<16} {n}")

    # --------------------------------------------------- dividir 80/20
    print("\n" + "=" * 74)
    print("PASO 4: DIVISION ESTRATIFICADA")
    print("=" * 74)

    for clase in CLASES:
        nombres = list(guardados[clase])
        random.shuffle(nombres)
        n_val = max(1, int(round(len(nombres) * args.frac_val)))
        n_val = min(n_val, max(1, len(nombres) - 1))

        for nombre in nombres[:n_val]:
            shutil.move(str(destino / nombre), str(destino / "val" / f"kia_{clase}" / nombre))
        for nombre in nombres[n_val:]:
            shutil.move(str(destino / nombre), str(destino / "train" / f"kia_{clase}" / nombre))

        print(f"  {clase:<12} train={len(nombres) - n_val:<5} val={n_val}")

    # --------------------------------------------------- data.yaml
    rutas_yaml = [str(destino.resolve()).replace('\\', '/')]
    contenido = (
        "# Dataset de clasificacion marca/modelo Kia (reconstruido)\n"
        f"path: {rutas_yaml[0]}\n"
        "train: train\n"
        "val: val\n"
        "names:\n"
    )
    for indice, clase in enumerate(CLASES):
        contenido += f"  {indice}: {clase}\n"
    (destino / "data.yaml").write_text(contenido, encoding='utf-8')

    # --------------------------------------------------- resumen final
    print("\n" + "=" * 74)
    print("RESUMEN FINAL")
    print("=" * 74)
    total_train = total_val = 0
    print(f"{'clase':<12} {'train':>7} {'val':>6} {'total':>7}")
    for clase in CLASES:
        n_train = len(list((destino / 'train' / f'kia_{clase}').glob('*.jpg')))
        n_val = len(list((destino / 'val' / f'kia_{clase}').glob('*.jpg')))
        total_train += n_train
        total_val += n_val
        marca = "  <-- REVISAR" if n_train + n_val < 120 else ""
        print(f"{clase:<12} {n_train:>7} {n_val:>6} {n_train + n_val:>7}{marca}")

    print("-" * 74)
    print(f"{'TOTAL':<12} {total_train:>7} {total_val:>6} {total_train + total_val:>7}")
    print("=" * 74)
    print(f"\nDataset listo en: {destino}")
    print(f"data.yaml:       {destino / 'data.yaml'}")
    print("\nSiguiente paso (lo ejecutas tu):")
    print(f"  python entrenar_clasificador_v2.py")


if __name__ == "__main__":
    main()
