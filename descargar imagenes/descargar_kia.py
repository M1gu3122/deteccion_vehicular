import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import hf_hub_download, list_repo_files


REPO = "venetis/VMMRdb_make_model_train"
API = f"https://huggingface.co/api/datasets/{REPO}"

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_CLASES = BASE_DIR / "modelos" / "vmmrdb_clases.json"


def obtener_nombres():
    """Devuelve {indice: nombre_clase} del dataset (con cache local)."""
    if CACHE_CLASES.exists():
        with CACHE_CLASES.open("r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
        return {int(indice): nombre for indice, nombre in datos.items()}

    respuesta = requests.get(API, timeout=30)
    respuesta.raise_for_status()
    datos = respuesta.json()

    for feature in datos["cardData"]["dataset_info"]["features"]:
        if feature["name"] == "label":
            nombres = feature["dtype"]["class_label"]["names"]
            CACHE_CLASES.parent.mkdir(parents=True, exist_ok=True)
            with CACHE_CLASES.open("w", encoding="utf-8") as archivo:
                json.dump(nombres, archivo, ensure_ascii=False, indent=2)
            return {int(indice): nombre for indice, nombre in nombres.items()}

    raise RuntimeError("No se encontraron las clases del dataset.")


def seleccionar_clases(nombres, filtro):
    seleccion = {}
    for indice, nombre in nombres.items():
        if not nombre.lower().startswith("kia"):
            continue
        if filtro and nombre not in filtro:
            continue
        seleccion[int(indice)] = nombre
    return seleccion


def bytes_de_imagen(celda):
    if isinstance(celda, dict):
        return celda.get("bytes")
    if isinstance(celda, (bytes, bytearray)):
        return bytes(celda)
    return None


def nombre_archivo(clase, datos):
    """Nombre estable por contenido: re-procesar no duplica imagenes."""
    firma = hashlib.sha1(datos).hexdigest()[:16]
    return f"{clase}_{firma}.jpg"


def guardar_imagen(datos, destino):
    if not datos:
        return False

    arreglo = np.frombuffer(datos, dtype=np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)
    if imagen is None:
        return False

    return bool(cv2.imwrite(str(destino), imagen))


def contar_existentes(salida, clases):
    """Cuenta imagenes ya guardadas para reanudar sin sobrescribir."""
    contador = {}
    for nombre in clases.values():
        total = 0
        for particion in ("train", "val"):
            carpeta = Path(salida) / particion / nombre
            if carpeta.is_dir():
                total += sum(1 for _ in carpeta.glob("*.jpg"))
        contador[nombre] = total
    return contador


def procesar_parquet(ruta, clases, salida, maximo, contador):
    """Recorre un parquet y guarda solo las imagenes de las clases pedidas."""
    total = 0
    parquet = pq.ParquetFile(str(ruta))

    for lote in parquet.iter_batches(batch_size=200, columns=["image", "label"]):
        etiquetas = lote.column("label").to_pylist()
        imagenes = lote.column("image").to_pylist()

        for etiqueta, celda in zip(etiquetas, imagenes):
            if etiqueta is None:
                continue

            indice = int(etiqueta)
            if indice not in clases:
                continue

            clase = clases[indice]

            datos = bytes_de_imagen(celda)
            if not datos:
                continue

            nombre = nombre_archivo(clase, datos)
            carpeta_train = Path(salida) / "train" / clase
            carpeta_val = Path(salida) / "val" / clase

            if (carpeta_train / nombre).exists() or (carpeta_val / nombre).exists():
                continue

            if maximo and contador[clase] >= maximo:
                continue

            particion = "val" if contador[clase] % 5 == 4 else "train"
            carpeta = Path(salida) / particion / clase
            carpeta.mkdir(parents=True, exist_ok=True)

            if guardar_imagen(datos, carpeta / nombre):
                contador[clase] += 1
                total += 1

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Descarga solo las clases Kia de VMMRdb (mirror HuggingFace)."
    )
    parser.add_argument(
        "--salida",
        default=str(BASE_DIR / "dataset_autos"),
        help="Carpeta raiz del dataset (crea train/ y val/).",
    )
    parser.add_argument(
        "--clases",
        default=None,
        help="Lista separada por comas, p.ej. kia_rio,kia_sportage.",
    )
    parser.add_argument(
        "--max-por-clase",
        type=int,
        default=400,
        help="Maximo de imagenes por clase (0 = sin limite).",
    )
    parser.add_argument(
        "--shards",
        default=None,
        help="Indices de shards a procesar, p.ej. 4,5,6 (por defecto todos).",
    )
    parser.add_argument(
        "--borrar-parquet",
        action="store_true",
        help="Borra cada parquet despues de procesarlo (ahorra disco).",
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Solo lista las clases Kia disponibles y sale.",
    )
    args = parser.parse_args()

    nombres = obtener_nombres()

    filtro = None
    if args.clases:
        filtro = {c.strip() for c in args.clases.split(",") if c.strip()}

    clases = seleccionar_clases(nombres, filtro)

    if not clases:
        print("No hay clases Kia que coincidan con el filtro.")
        return

    if args.listar:
        for indice, nombre in sorted(clases.items(), key=lambda x: x[1]):
            print(f"  {indice:>3}  {nombre}")
        return

    print("Clases Kia seleccionadas:")
    for nombre in sorted(clases.values()):
        print(f"  - {nombre}")

    archivos = [
        archivo
        for archivo in list_repo_files(REPO, repo_type="dataset")
        if archivo.endswith(".parquet")
    ]
    archivos.sort()

    if args.shards:
        indices = {int(s) for s in args.shards.split(",") if s.strip()}
        archivos = [
            archivo
            for posicion, archivo in enumerate(archivos)
            if posicion in indices
        ]

    maximo = args.max_por_clase or None
    contador = contar_existentes(args.salida, clases)
    total = 0

    if any(contador.values()):
        print("Reanudando desde imagenes existentes:", contador)

    for archivo in archivos:
        if maximo and all(valor >= maximo for valor in contador.values()):
            break

        print(f"Descargando {archivo} ...")
        ruta = hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=archivo,
        )

        print(f"Procesando {archivo} ...")
        total += procesar_parquet(ruta, clases, args.salida, maximo, contador)

        if args.borrar_parquet:
            try:
                Path(ruta).unlink()
            except OSError as error:
                print(f"No se pudo borrar {ruta}: {error}")

    print("\nResumen:")
    for nombre, cantidad in sorted(contador.items()):
        print(f"  {nombre}: {cantidad}")
    print(f"Total guardado: {total} imagenes en {args.salida}")

    faltantes = [nombre for nombre, cantidad in contador.items() if cantidad == 0]
    if faltantes:
        print("\nSin imagenes (prueba con otros --shards):", ", ".join(sorted(faltantes)))


if __name__ == "__main__":
    main()
