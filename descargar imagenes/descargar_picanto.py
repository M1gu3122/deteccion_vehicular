"""
Descarga imágenes de Kia Picanto en Colombia usando Serper.

Uso en PowerShell:
    $env:SERPER_API_KEY="tu_api_key"
    python descargar_picanto.py --total 100

Opcional:
    python descargar_picanto.py --total 100 --paginas 2
    python descargar_picanto.py --total 20 --ancho-min 640 --alto-min 420

Las imágenes se guardan en:
    dataset_autos/picanto/

Los metadatos se registran en:
    dataset_autos/picanto/metadata.jsonl

Nota:
- Respeta las condiciones de uso y licencias de las páginas de origen.
- Las URLs originales pueden no permitir la descarga directa.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parent

PROVEEDORES = {
    "serper": {
        "url": "https://google.serper.dev/images",
        "clave": "aeadd1a916be809ce2d52c04d83fd46c5b4ce84c",
        "respuesta": "images",
        "pagina_inicial": 1,
    },
}

CONSULTAS = [
    ("picanto_2012", "Kia Picanto 2012 Colombia"),
    ("picanto_2014", "Kia Picanto 2014 Colombia"),
    ("picanto_2016", "Kia Picanto 2016 Colombia"),
    ("picanto_2018", "Kia Picanto 2018 Colombia"),
    ("picanto_2020", "Kia Picanto 2020 Colombia"),
    ("picanto_2022", "Kia Picanto 2022 Colombia"),
    ("picanto_2024", "Kia Picanto 2024 Colombia"),
    ("frontal", "Kia Picanto frontal Colombia"),
    ("lateral", "Kia Picanto lateral Colombia"),
    ("trasera", "Kia Picanto trasera Colombia"),
    ("tres_cuartos", "Kia Picanto tres cuartos Colombia"),
    ("interior", "Kia Picanto interior tablero Colombia"),
    ("gt_line", "Kia Picanto GT Line Colombia"),
    ("placa_colombiana", "Kia Picanto placa colombiana"),
]

CABECERAS_DESCARGA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

TIMEOUT_BUSQUEDA = 40
TIMEOUT_DESCARGA = 25
TAMANO_MINIMO_BYTES = 15_000
TAMANO_MAXIMO_BYTES = 25_000_000


def limpiar(texto):
    """Convierte una etiqueta en un nombre de archivo seguro."""
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_").lower()
    return texto[:40] or "img"


def parametros_busqueda(proveedor, consulta, pagina):
    """Construye los parámetros específicos de Serper."""
    if proveedor == "serper":
        return {
            "q": consulta,
            "gl": "co",
            "hl": "es",
            "page": pagina,
        }

    raise ValueError(f"Proveedor no soportado: {proveedor}")


def imagen_de_resultado(resultado):
    """Obtiene la URL de imagen original, con miniatura como alternativa."""
    return resultado.get("imageUrl") or resultado.get("thumbnailUrl")


def fuente_de_resultado(resultado):
    """Obtiene el nombre del sitio y la URL de la página de origen."""
    fuente = resultado.get("source", "")
    enlace = resultado.get("link", "")

    if isinstance(fuente, dict):
        return fuente.get("name", ""), fuente.get("link", enlace)

    return str(fuente or ""), str(enlace or "")


def buscar(proveedor, api_key, consulta, pagina, timeout=TIMEOUT_BUSQUEDA):
    """Consulta Serper y devuelve sus resultados de imágenes."""
    config = PROVEEDORES[proveedor]
    parametros = parametros_busqueda(proveedor, consulta, pagina)
    cabeceras = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    for intento in range(3):
        try:
            respuesta = requests.post(
                config["url"],
                headers=cabeceras,
                json=parametros,
                timeout=timeout,
            )

            if respuesta.status_code == 429:
                espera = 5 * (intento + 1)
                print(f"  Límite de consultas (429). Esperando {espera} s...")
                time.sleep(espera)
                continue

            if respuesta.status_code in (401, 403):
                print(
                    "  Error de autenticación o permisos. "
                    "Comprueba SERPER_API_KEY y tu plan."
                )
                return []

            respuesta.raise_for_status()
            datos = respuesta.json()

            if datos.get("error"):
                print(f"  Error de API: {datos['error']}")
                return []

            resultados = datos.get(config["respuesta"], [])
            if not isinstance(resultados, list):
                print("  La API devolvió un formato de resultados inesperado.")
                return []

            return resultados

        except (requests.RequestException, ValueError) as error:
            print(f"  Error en búsqueda (intento {intento + 1}/3): {error}")
            if intento < 2:
                time.sleep(3 * (intento + 1))

    return []


def url_segura(url):
    """Acepta únicamente enlaces HTTP o HTTPS bien formados."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        return False


def descargar(url, timeout=TIMEOUT_DESCARGA):
    """Descarga una imagen, validando respuesta, tipo y tamaño."""
    if not url_segura(url):
        return None

    for intento in range(3):
        try:
            with requests.get(
                url,
                headers=CABECERAS_DESCARGA,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
            ) as respuesta:
                if respuesta.status_code == 429 or respuesta.status_code >= 500:
                    time.sleep(1 + intento)
                    continue

                if respuesta.status_code != 200:
                    return None

                tipo = respuesta.headers.get("Content-Type", "").lower()
                if "image/" not in tipo:
                    return None

                contenido = bytearray()
                for bloque in respuesta.iter_content(chunk_size=64 * 1024):
                    if not bloque:
                        continue
                    contenido.extend(bloque)
                    if len(contenido) > TAMANO_MAXIMO_BYTES:
                        return None

                if len(contenido) < TAMANO_MINIMO_BYTES:
                    return None

                return bytes(contenido)

        except requests.RequestException:
            if intento < 2:
                time.sleep(1 + intento)

    return None


def hash_contenido(datos):
    """Hash SHA-256 para identificar archivos idénticos."""
    return hashlib.sha256(datos).hexdigest()


def hash_visual(imagen):
    """
    Hash perceptual sencillo de 64 bits.
    Se utiliza para detectar imágenes muy parecidas, no como prueba definitiva.
    """
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    reducido = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)
    diferencias = reducido[:, 1:] > reducido[:, :-1]
    return "".join("1" if bit else "0" for bit in diferencias.flatten())


def distancia_hamming(hash_a, hash_b):
    """Cuenta los bits diferentes entre dos hashes perceptuales."""
    if len(hash_a) != len(hash_b):
        return max(len(hash_a), len(hash_b))
    return sum(a != b for a, b in zip(hash_a, hash_b))


def procesar(datos, ancho_min, alto_min):
    """Decodifica una imagen y verifica sus dimensiones mínimas."""
    arreglo = np.frombuffer(datos, dtype=np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

    if imagen is None:
        return None

    alto, ancho = imagen.shape[:2]
    if ancho < ancho_min or alto < alto_min:
        return None

    return imagen


def leer_metadata(salida):
    """Lee los metadatos existentes y omite líneas JSON inválidas."""
    registros = []
    ruta = salida / "metadata.jsonl"

    if not ruta.exists():
        return registros

    with ruta.open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            try:
                registro = json.loads(linea)
                if isinstance(registro, dict):
                    registros.append(registro)
            except json.JSONDecodeError:
                continue

    return registros


def cargar_existentes(salida):
    """
    Carga hashes de metadata y calcula hashes de imágenes presentes en disco.
    Devuelve hashes de contenido, hashes visuales y nombres ya registrados.
    """
    hashes_contenido = set()
    hashes_visuales = []
    nombres_registrados = set()

    for registro in leer_metadata(salida):
        if registro.get("hash"):
            hashes_contenido.add(registro["hash"])
        if registro.get("hash_visual"):
            hashes_visuales.append(registro["hash_visual"])
        if registro.get("archivo"):
            nombres_registrados.add(registro["archivo"])

    # Revisa todos los archivos de imagen existentes, no solamente JPG.
    extensiones = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for archivo in salida.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in extensiones:
            try:
                datos = archivo.read_bytes()
                hashes_contenido.add(hash_contenido(datos))
                if archivo.name not in nombres_registrados:
                    imagen = cv2.imdecode(
                        np.frombuffer(datos, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if imagen is not None:
                        hashes_visuales.append(hash_visual(imagen))
            except OSError:
                continue

    return hashes_contenido, hashes_visuales, nombres_registrados


def guardar_metadata(salida, registro):
    """Añade un registro JSON por línea."""
    ruta = salida / "metadata.jsonl"
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Descarga imágenes de Kia Picanto en Colombia usando Serper."
    )
    parser.add_argument(
        "--proveedor",
        choices=sorted(PROVEEDORES),
        default="serper",
        help="Proveedor de búsqueda (por defecto: serper).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Si se omite, se lee desde SERPER_API_KEY.",
    )
    parser.add_argument(
        "--salida",
        default=str(BASE_DIR / "dataset_autos" / "picanto"),
        help="Carpeta de destino.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Cantidad objetivo total de imágenes.",
    )
    parser.add_argument(
        "--por-consulta",
        type=int,
        default=14,
        help="Máximo de candidatos por consulta y página.",
    )
    parser.add_argument(
        "--paginas",
        type=int,
        default=1,
        help="Número de páginas por consulta.",
    )
    parser.add_argument("--ancho-min", type=int, default=640)
    parser.add_argument("--alto-min", type=int, default=420)
    parser.add_argument(
        "--distancia-dup",
        type=int,
        default=4,
        help="Distancia Hamming máxima para marcar posible duplicado visual.",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=0.4,
        help="Pausa entre consultas en segundos.",
    )
    args = parser.parse_args()

    if args.total < 1:
        parser.error("--total debe ser mayor que 0.")
    if args.paginas < 1:
        parser.error("--paginas debe ser mayor que 0.")
    if args.por_consulta < 1:
        parser.error("--por-consulta debe ser mayor que 0.")
    if args.ancho_min < 1 or args.alto_min < 1:
        parser.error("Las dimensiones mínimas deben ser mayores que 0.")
    if not 0 <= args.distancia_dup <= 64:
        parser.error("--distancia-dup debe estar entre 0 y 64.")

    config = PROVEEDORES[args.proveedor]
    api_key = args.api_key or os.environ.get(config["clave"])

    if not api_key:
        raise SystemExit(
            f"Falta la API key. Define la variable de entorno "
            f"{config['clave']} o usa --api-key."
        )

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    hashes_contenido, hashes_visuales, nombres_registrados = cargar_existentes(
        salida
    )

    # Cuenta imágenes que ya están en el directorio. No cuenta líneas de metadata.
    extensiones = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    archivos_existentes = [
        archivo
        for archivo in salida.iterdir()
        if archivo.is_file() and archivo.suffix.lower() in extensiones
    ]
    guardadas = len(archivos_existentes)

    if guardadas >= args.total:
        print(f"Ya hay {guardadas} imágenes en {salida}.")
        return

    print(f"Objetivo: {args.total} imágenes (ya hay {guardadas}).")
    print(f"Proveedor: {args.proveedor}")
    print(f"Destino: {salida.resolve()}")

    candidatos_por_consulta = []

    for etiqueta, consulta in CONSULTAS:
        candidatos_consulta = []

        for pagina in range(
            config["pagina_inicial"],
            config["pagina_inicial"] + args.paginas,
        ):
            print(f"Buscando: {consulta} (página {pagina})")
            resultados = buscar(
                args.proveedor,
                api_key,
                consulta,
                pagina,
            )
            print(f"  {len(resultados)} resultados recibidos")

            for resultado in resultados[: args.por_consulta]:
                url = imagen_de_resultado(resultado)
                if not url_segura(url):
                    continue

                nombre_fuente, enlace_fuente = fuente_de_resultado(resultado)
                candidatos_consulta.append(
                    {
                        "url": url,
                        "etiqueta": etiqueta,
                        "consulta": consulta,
                        "titulo": resultado.get("title", ""),
                        "fuente": nombre_fuente,
                        "enlace": enlace_fuente,
                    }
                )

            time.sleep(max(0, args.pausa))

        if candidatos_consulta:
            candidatos_por_consulta.append(candidatos_consulta)

    if not candidatos_por_consulta:
        raise SystemExit(
            "No se obtuvieron resultados. Comprueba la API key, el plan "
            "de Serper y la estructura de respuesta."
        )

    # Intercala resultados de las consultas para aumentar la variedad.
    intercalados = []
    max_len = max(len(lista) for lista in candidatos_por_consulta)
    for posicion in range(max_len):
        for lista in candidatos_por_consulta:
            if posicion < len(lista):
                intercalados.append(lista[posicion])

    urls_vistas = set()
    descargadas = 0
    fallidas = 0
    duplicadas = 0
    descartadas = 0

    for candidato in intercalados:
        if guardadas >= args.total:
            break

        url = candidato["url"]
        if url in urls_vistas:
            duplicadas += 1
            continue
        urls_vistas.add(url)

        datos = descargar(url)
        if not datos:
            fallidas += 1
            continue

        firma = hash_contenido(datos)
        if firma in hashes_contenido:
            duplicadas += 1
            continue

        imagen = procesar(datos, args.ancho_min, args.alto_min)
        if imagen is None:
            descartadas += 1
            continue

        visual = hash_visual(imagen)
        if any(
            distancia_hamming(visual, previo) <= args.distancia_dup
            for previo in hashes_visuales
        ):
            duplicadas += 1
            continue

        nombre = f"{limpiar(candidato['etiqueta'])}_{firma[:16]}.jpg"
        destino = salida / nombre

        # Evita sobrescribir un archivo si ya existe por cualquier motivo.
        if destino.exists():
            duplicadas += 1
            continue

        if not cv2.imwrite(str(destino), imagen):
            fallidas += 1
            continue

        registro = {
            "archivo": nombre,
            "hash": firma,
            "hash_visual": visual,
            "etiqueta": candidato["etiqueta"],
            "consulta": candidato["consulta"],
            "titulo": candidato["titulo"],
            "fuente": candidato["fuente"],
            "url_imagen": candidato["url"],
            "url_pagina": candidato["enlace"],
            "ancho": int(imagen.shape[1]),
            "alto": int(imagen.shape[0]),
            "proveedor_busqueda": args.proveedor,
            "fecha_descarga": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        guardar_metadata(salida, registro)

        hashes_contenido.add(firma)
        hashes_visuales.append(visual)
        nombres_registrados.add(nombre)
        guardadas += 1
        descargadas += 1

        print(f"  [{guardadas}/{args.total}] Guardada: {nombre}")

    print(
        "\nProceso terminado:"
        f"\n  Imágenes presentes: {guardadas}"
        f"\n  Nuevas descargas: {descargadas}"
        f"\n  Descargas fallidas: {fallidas}"
        f"\n  Descartadas por formato o dimensiones: {descartadas}"
        f"\n  Duplicadas o muy similares: {duplicadas}"
        f"\n  Carpeta: {salida.resolve()}"
        f"\n  Metadatos: {(salida / 'metadata.jsonl').resolve()}"
    )

    if guardadas < args.total:
        print(
            "\nNo se alcanzó el objetivo. Puedes intentar con más páginas "
            "(--paginas 2) o revisar los filtros de tamaño y duplicados."
        )


if __name__ == "__main__":
    main()
