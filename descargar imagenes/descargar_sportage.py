# -*- coding: utf-8 -*-
"""
Descargador de imágenes de Kia Sportage mediante Serper.dev.
Solo busca exteriores. Requiere revisión manual antes de etiquetar el dataset.
"""

import os
import re
import cv2
import json
import time
import hashlib
import argparse
import requests
import numpy as np

from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CARPETA_SALIDA = BASE_DIR / "dataset_autos" / "sportage"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

METADATA_FILE = CARPETA_SALIDA / "metadata.jsonl"

SERPER_URL = "https://google.serper.dev/images"

ANCHO_MINIMO = 640
ALTO_MINIMO = 420
BYTES_MINIMOS = 15 * 1024
BYTES_MAXIMOS = 25 * 1024 * 1024
DISTANCIA_DUPLICADOS = 5

TIMEOUT = 25
REINTENTOS = 3


# ============================================================
# CONSULTAS DE BÚSQUEDA: SOLO EXTERIORES
# ============================================================

CONSULTAS = [
    ('sportage_1995', 'Kia Sportage 1995 exterior'),
    ('sportage_1996', 'Kia Sportage 1996 exterior'),
    ('sportage_1997', 'Kia Sportage 1997 exterior'),
    ('sportage_1998', 'Kia Sportage 1998 exterior'),
    ('sportage_1999', 'Kia Sportage 1999 exterior'),
    ('sportage_2000', 'Kia Sportage 2000 exterior'),
    ('sportage_2001', 'Kia Sportage 2001 exterior'),
    ('sportage_2002', 'Kia Sportage 2002 exterior'),
    ('sportage_2003', 'Kia Sportage 2003 exterior'),
    ('sportage_2004', 'Kia Sportage 2004 exterior'),
    ('sportage_2005', 'Kia Sportage 2005 exterior'),
    ('sportage_2006', 'Kia Sportage 2006 exterior'),
    ('sportage_2007', 'Kia Sportage 2007 exterior'),
    ('sportage_2008', 'Kia Sportage 2008 exterior'),
    ('sportage_2009', 'Kia Sportage 2009 exterior'),
    ('sportage_2010', 'Kia Sportage 2010 exterior'),
    ('sportage_2011', 'Kia Sportage 2011 exterior'),
    ('sportage_2012', 'Kia Sportage 2012 exterior'),
    ('sportage_2013', 'Kia Sportage 2013 exterior'),
    ('sportage_2014', 'Kia Sportage 2014 exterior'),
    ('sportage_2015', 'Kia Sportage 2015 exterior'),
    ('sportage_2016', 'Kia Sportage 2016 exterior'),
    ('sportage_2017', 'Kia Sportage 2017 exterior'),
    ('sportage_2018', 'Kia Sportage 2018 exterior'),
    ('sportage_2019', 'Kia Sportage 2019 exterior'),
    ('sportage_2020', 'Kia Sportage 2020 exterior'),
    ('sportage_2021', 'Kia Sportage 2021 exterior'),
    ('sportage_2022', 'Kia Sportage 2022 exterior'),
    ('sportage_2023', 'Kia Sportage 2023 exterior'),
    ('sportage_2024', 'Kia Sportage 2024 exterior'),
    ('sportage_2025', 'Kia Sportage 2025 exterior'),
    ('sportage_2026', 'Kia Sportage 2026 exterior'),
    ('primera_generacion', 'Kia Sportage first generation exterior'),
    ('segunda_generacion', 'Kia Sportage second generation exterior'),
    ('tercera_generacion', 'Kia Sportage third generation exterior'),
    ('cuarta_generacion', 'Kia Sportage fourth generation exterior'),
    ('quinta_generacion', 'Kia Sportage fifth generation exterior'),
    ('sportage_nb7', 'Kia Sportage NB7 exterior'),
    ('sportage_km', 'Kia Sportage KM exterior'),
    ('sportage_sl', 'Kia Sportage SL exterior'),
    ('sportage_ql', 'Kia Sportage QL exterior'),
    ('sportage_nq5', 'Kia Sportage NQ5 exterior'),
    ('sportage_lx', 'Kia Sportage LX exterior'),
    ('sportage_ex', 'Kia Sportage EX exterior'),
    ('sportage_sx', 'Kia Sportage SX exterior'),
    ('sportage_gt_line', 'Kia Sportage GT Line exterior'),
    ('sportage_x_line', 'Kia Sportage X-Line exterior'),
    ('sportage_x_pro', 'Kia Sportage X-Pro exterior'),
    ('sportage_hybrid', 'Kia Sportage Hybrid exterior'),
    ('sportage_phev', 'Kia Sportage Plug-in Hybrid exterior'),
    ('sportage_diesel', 'Kia Sportage diesel exterior'),
    ('sportage_awd', 'Kia Sportage AWD exterior'),
    ('frontal', 'Kia Sportage front view exterior'),
    ('trasera', 'Kia Sportage rear view exterior'),
    ('lateral', 'Kia Sportage side view exterior'),
    ('tres_cuartos_frontal', 'Kia Sportage front three quarter view exterior'),
    ('tres_cuartos_trasero', 'Kia Sportage rear three quarter view exterior'),
    ('parqueadero', 'Kia Sportage parked in parking lot exterior'),
    ('calle', 'Kia Sportage driving on city street exterior'),
    ('concesionario', 'Kia Sportage exterior car dealership'),
    ('carretera', 'Kia Sportage driving on road exterior'),
    ('luz_dia', 'Kia Sportage exterior daylight photography'),
    ('noche', 'Kia Sportage exterior at night'),
    ('lluvia', 'Kia Sportage exterior rainy weather'),
    ('varios_vehiculos', 'Kia Sportage parked among other cars exterior'),
    ('sportage_colombia', 'Kia Sportage Colombia fotografías exterior'),
    ('sportage_usado_colombia', 'Kia Sportage usado Colombia exterior'),
    ('sportage_placa_colombiana', 'Kia Sportage con placa colombiana exterior'),
]


# ============================================================
# UTILIDADES
# ============================================================

def sha256_bytes(contenido):
    return hashlib.sha256(contenido).hexdigest()


def limpiar_nombre(texto):
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9_-]+", "_", texto)
    return texto.strip("_")


def hash_perceptual(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    reducido = cv2.resize(gris, (8, 8), interpolation=cv2.INTER_AREA)
    promedio = reducido.mean()
    bits = reducido > promedio

    valor = 0
    for bit in bits.flatten():
        valor = (valor << 1) | int(bit)

    return valor


def distancia_hash(hash1, hash2):
    return bin(hash1 ^ hash2).count("1")


def cargar_hashes_existentes():
    hashes_sha = set()
    hashes_visual = []

    for archivo in CARPETA_SALIDA.iterdir():
        if archivo.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        try:
            contenido = archivo.read_bytes()
            imagen = cv2.imdecode(
                np.frombuffer(contenido, np.uint8),
                cv2.IMREAD_COLOR
            )

            if imagen is None:
                continue

            hashes_sha.add(sha256_bytes(contenido))
            hashes_visual.append(hash_perceptual(imagen))

        except Exception as error:
            print(f"[AVISO] No se pudo leer {archivo.name}: {error}")

    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                try:
                    dato = json.loads(linea)
                    if dato.get("sha256"):
                        hashes_sha.add(dato["sha256"])
                except json.JSONDecodeError:
                    continue

    return hashes_sha, hashes_visual


# ============================================================
# SERPER.DEV
# ============================================================

def buscar(consulta, pagina, api_key):
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    parametros = {
        "q": consulta,
        "gl": "co",
        "hl": "es",
        "page": pagina,
    }

    for intento in range(REINTENTOS):
        try:
            respuesta = requests.post(
                SERPER_URL,
                headers=headers,
                json=parametros,
                timeout=TIMEOUT
            )

            if respuesta.status_code == 429:
                print("[AVISO] Límite de solicitudes alcanzado.")
                time.sleep(5 * (intento + 1))
                continue

            if respuesta.status_code in [401, 403]:
                raise RuntimeError(
                    "API Key inválida o sin permisos en Serper.dev."
                )

            respuesta.raise_for_status()
            return respuesta.json().get("images", [])

        except requests.exceptions.Timeout:
            print("[AVISO] Tiempo de espera agotado.")

        except requests.exceptions.RequestException as error:
            print(f"[ERROR] Error de búsqueda: {error}")

        if intento < REINTENTOS - 1:
            time.sleep(2)

    return []


def imagen_de_resultado(resultado):
    return resultado.get("imageUrl") or resultado.get("thumbnailUrl")


def fuente_de_resultado(resultado):
    fuente = resultado.get("source", "")

    if isinstance(fuente, dict):
        return fuente.get("name") or fuente.get("link") or ""

    return str(fuente)


# ============================================================
# DESCARGA Y VALIDACIÓN
# ============================================================

def descargar_imagen(url):
    if not url:
        return None

    try:
        respuesta = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        respuesta.raise_for_status()

        contenido = respuesta.content

        if not (BYTES_MINIMOS <= len(contenido) <= BYTES_MAXIMOS):
            return None

        imagen = cv2.imdecode(
            np.frombuffer(contenido, np.uint8),
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            return None

        alto, ancho = imagen.shape[:2]

        if ancho < ANCHO_MINIMO or alto < ALTO_MINIMO:
            return None

        return contenido, imagen

    except Exception:
        return None


# ============================================================
# GUARDAR IMAGEN Y METADATOS
# ============================================================

def guardar_imagen(
    contenido,
    imagen,
    consulta_id,
    resultado,
    hashes_sha,
    hashes_visual,
    siguiente_numero
):
    hash_sha = sha256_bytes(contenido)

    if hash_sha in hashes_sha:
        return False

    hash_visual = hash_perceptual(imagen)

    for existente in hashes_visual:
        if distancia_hash(hash_visual, existente) <= DISTANCIA_DUPLICADOS:
            return False

    nombre_archivo = (
        f"sportage_{limpiar_nombre(consulta_id)}_{siguiente_numero:05d}.jpg"
    )
    ruta = CARPETA_SALIDA / nombre_archivo

    exito, codificado = cv2.imencode(
        ".jpg",
        imagen,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not exito:
        return False

    ruta.write_bytes(codificado.tobytes())

    hashes_sha.add(hash_sha)
    hashes_visual.append(hash_visual)

    metadato = {
        "archivo": nombre_archivo,
        "modelo": "sportage",
        "consulta": consulta_id,
        "url_imagen": imagen_de_resultado(resultado),
        "url_origen": resultado.get("link", ""),
        "fuente": fuente_de_resultado(resultado),
        "ancho": int(imagen.shape[1]),
        "alto": int(imagen.shape[0]),
        "sha256": hash_sha,
        "fecha_descarga": datetime.now().isoformat(),
    }

    with open(METADATA_FILE, "a", encoding="utf-8") as archivo:
        archivo.write(
            json.dumps(metadato, ensure_ascii=False) + "\n"
        )

    return True


# ============================================================
# INTERCALAR RESULTADOS
# ============================================================

def intercalar_consultas(candidatos):
    resultado = []
    maximo = max((len(lista) for lista in candidatos.values()), default=0)

    for posicion in range(maximo):
        for consulta_id, lista in candidatos.items():
            if posicion < len(lista):
                resultado.append((consulta_id, lista[posicion]))

    return resultado


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def ejecutar(args):
    api_key = args.api_key or os.getenv("SERPER_API_KEY")

    if api_key:
        api_key = api_key.strip()

    if not api_key:
        raise RuntimeError(
            "Falta la API Key. Utiliza --api-key o configura SERPER_API_KEY."
        )

    hashes_sha, hashes_visual = cargar_hashes_existentes()

    print("\\n======================================")
    print("       DESCARGADOR KIA SPORTAGE")
    print("======================================")
    print(f"Imágenes existentes: {len(hashes_sha)}")
    print(f"Objetivo acumulado: {args.total}")
    print(f"Consultas: {len(CONSULTAS)}")
    print(f"Páginas por consulta: {args.paginas}")

    if len(hashes_sha) >= args.total:
        print("\\nYa se alcanzó el objetivo de imágenes.")
        return

    candidatos = {}

    # FASE 1: buscar imágenes
    for consulta_id, consulta in CONSULTAS:
        candidatos[consulta_id] = []
        print(f"\\n[BUSCANDO] {consulta}")

        for pagina in range(
            args.pagina_inicial,
            args.pagina_inicial + args.paginas
        ):
            resultados = buscar(consulta, pagina, api_key)

            print(f"   Página {pagina}: {len(resultados)} resultados")

            for resultado in resultados:
                if imagen_de_resultado(resultado):
                    candidatos[consulta_id].append(resultado)

            time.sleep(args.pausa)

    # FASE 2: intercalar resultados
    lista_final = intercalar_consultas(candidatos)

    print("\\n======================================")
    print("COMENZANDO DESCARGA")
    print("======================================")

    descargadas = 0
    siguiente_numero = len(hashes_sha) + 1

    # FASE 3: descargar y validar
    for consulta_id, resultado in lista_final:
        if len(hashes_sha) >= args.total:
            break

        url = imagen_de_resultado(resultado)
        descargada = descargar_imagen(url)

        if descargada is None:
            continue

        contenido, imagen = descargada

        guardada = guardar_imagen(
            contenido,
            imagen,
            consulta_id,
            resultado,
            hashes_sha,
            hashes_visual,
            siguiente_numero
        )

        if guardada:
            descargadas += 1
            siguiente_numero += 1

            print(
                f"[OK] {descargadas} nuevas | "
                f"Total: {len(hashes_sha)} | {consulta_id}"
            )
        else:
            print(f"[DUPLICADA] {consulta_id}")

        time.sleep(args.pausa)

    print("\\n======================================")
    print("DESCARGA FINALIZADA")
    print("======================================")
    print(f"Nuevas imágenes: {descargadas}")
    print(f"Total acumulado: {len(hashes_sha)}")
    print(f"Carpeta: {CARPETA_SALIDA}")
    print(f"Metadatos: {METADATA_FILE}")


# ============================================================
# ARGUMENTOS DE CONSOLA
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Descargador de imágenes Kia Sportage con Serper.dev"
    )

    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--paginas", type=int, default=1)
    parser.add_argument("--pagina-inicial", type=int, default=1)
    parser.add_argument("--pausa", type=float, default=1.0)
    parser.add_argument("--ancho-min", type=int, default=640)
    parser.add_argument("--alto-min", type=int, default=420)
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key de Serper.dev"
    )

    args = parser.parse_args()

    if args.total < 1 or args.paginas < 1 or args.pagina_inicial < 1:
        parser.error("--total, --paginas y --pagina-inicial deben ser mayores que cero")

    if args.pausa < 0:
        parser.error("--pausa no puede ser negativa")

    global ANCHO_MINIMO
    global ALTO_MINIMO

    ANCHO_MINIMO = args.ancho_min
    ALTO_MINIMO = args.alto_min

    ejecutar(args)


if __name__ == "__main__":
    main()
