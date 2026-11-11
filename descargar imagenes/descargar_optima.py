# -*- coding: utf-8 -*-
"""
Descargador de imágenes de Kia Optima mediante Serper.dev.
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
CARPETA_SALIDA = BASE_DIR / "dataset_autos" / "optima"
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
    ('optima_2000', 'Kia Optima 2000 exterior'),
    ('optima_2001', 'Kia Optima 2001 exterior'),
    ('optima_2002', 'Kia Optima 2002 exterior'),
    ('optima_2003', 'Kia Optima 2003 exterior'),
    ('optima_2004', 'Kia Optima 2004 exterior'),
    ('optima_2005', 'Kia Optima 2005 exterior'),
    ('optima_2006', 'Kia Optima 2006 exterior'),
    ('optima_2007', 'Kia Optima 2007 exterior'),
    ('optima_2008', 'Kia Optima 2008 exterior'),
    ('optima_2009', 'Kia Optima 2009 exterior'),
    ('optima_2010', 'Kia Optima 2010 exterior'),
    ('optima_2011', 'Kia Optima 2011 exterior'),
    ('optima_2012', 'Kia Optima 2012 exterior'),
    ('optima_2013', 'Kia Optima 2013 exterior'),
    ('optima_2014', 'Kia Optima 2014 exterior'),
    ('optima_2015', 'Kia Optima 2015 exterior'),
    ('optima_2016', 'Kia Optima 2016 exterior'),
    ('optima_2017', 'Kia Optima 2017 exterior'),
    ('optima_2018', 'Kia Optima 2018 exterior'),
    ('optima_2019', 'Kia Optima 2019 exterior'),
    ('optima_2020', 'Kia Optima 2020 exterior'),
    ('primera_generacion', 'Kia Optima first generation exterior'),
    ('segunda_generacion', 'Kia Optima second generation exterior'),
    ('tercera_generacion', 'Kia Optima third generation exterior'),
    ('cuarta_generacion', 'Kia Optima fourth generation exterior'),
    ('quinta_generacion', 'Kia K5 fifth generation exterior'),
    ('optima_mg', 'Kia Optima MG exterior'),
    ('optima_tf', 'Kia Optima TF exterior'),
    ('optima_jf', 'Kia Optima JF exterior'),
    ('kia_k5_dl3', 'Kia K5 DL3 exterior'),
    ('optima_lx', 'Kia Optima LX exterior'),
    ('optima_ex', 'Kia Optima EX exterior'),
    ('optima_sx', 'Kia Optima SX exterior'),
    ('optima_sx_turbo', 'Kia Optima SX Turbo exterior'),
    ('optima_hybrid', 'Kia Optima Hybrid exterior'),
    ('optima_phev', 'Kia Optima Plug-in Hybrid exterior'),
    ('k5_gt_line', 'Kia K5 GT-Line exterior'),
    ('k5_gt', 'Kia K5 GT exterior'),
    ('frontal', 'Kia Optima front view exterior'),
    ('trasera', 'Kia Optima rear view exterior'),
    ('lateral', 'Kia Optima side view exterior'),
    ('tres_cuartos_frontal', 'Kia Optima front three quarter view exterior'),
    ('tres_cuartos_trasero', 'Kia Optima rear three quarter view exterior'),
    ('parqueadero', 'Kia Optima parked in parking lot exterior'),
    ('calle', 'Kia Optima driving on city street exterior'),
    ('concesionario', 'Kia Optima exterior car dealership'),
    ('carretera', 'Kia Optima driving on road exterior'),
    ('luz_dia', 'Kia Optima exterior daylight photography'),
    ('noche', 'Kia Optima exterior at night'),
    ('lluvia', 'Kia Optima exterior rainy weather'),
    ('varios_vehiculos', 'Kia Optima parked among other cars exterior'),
    ('optima_colombia', 'Kia Optima Colombia fotografías exterior'),
    ('optima_usado_colombia', 'Kia Optima usado Colombia exterior'),
    ('optima_placa_colombiana', 'Kia Optima con placa colombiana exterior'),
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
        f"optima_{limpiar_nombre(consulta_id)}_{siguiente_numero:05d}.jpg"
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
        "modelo": "optima",
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
    print("       DESCARGADOR KIA OPTIMA")
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
        description="Descargador de imágenes Kia Optima con Serper.dev"
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
