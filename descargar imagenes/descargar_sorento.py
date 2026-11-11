# descargar_sorento.py

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
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CARPETA_SALIDA = BASE_DIR / "dataset_autos" / "kia_sorento"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

METADATA_FILE = CARPETA_SALIDA / "metadata.jsonl"

PROVEEDORES = {
    "serper": {
        "url": "https://google.serper.dev/images",
        "clave": "SERPER_API_KEY",
        "respuesta": "images",
    }
}

PROVEEDOR_ACTIVO = "serper"


# ============================================================
# CONSULTAS: KIA SORENTO EXTERIOR
# ============================================================

CONSULTAS = [

    # --------------------------------------------------------
    # AÑOS
    # --------------------------------------------------------

    ("sorento_2002", "Kia Sorento 2002 exterior"),
    ("sorento_2003", "Kia Sorento 2003 exterior"),
    ("sorento_2004", "Kia Sorento 2004 exterior"),
    ("sorento_2005", "Kia Sorento 2005 exterior"),
    ("sorento_2006", "Kia Sorento 2006 exterior"),
    ("sorento_2007", "Kia Sorento 2007 exterior"),
    ("sorento_2008", "Kia Sorento 2008 exterior"),
    ("sorento_2009", "Kia Sorento 2009 exterior"),
    ("sorento_2010", "Kia Sorento 2010 exterior"),
    ("sorento_2011", "Kia Sorento 2011 exterior"),
    ("sorento_2012", "Kia Sorento 2012 exterior"),
    ("sorento_2013", "Kia Sorento 2013 exterior"),
    ("sorento_2014", "Kia Sorento 2014 exterior"),
    ("sorento_2015", "Kia Sorento 2015 exterior"),
    ("sorento_2016", "Kia Sorento 2016 exterior"),
    ("sorento_2017", "Kia Sorento 2017 exterior"),
    ("sorento_2018", "Kia Sorento 2018 exterior"),
    ("sorento_2019", "Kia Sorento 2019 exterior"),
    ("sorento_2020", "Kia Sorento 2020 exterior"),
    ("sorento_2021", "Kia Sorento 2021 exterior"),
    ("sorento_2022", "Kia Sorento 2022 exterior"),
    ("sorento_2023", "Kia Sorento 2023 exterior"),
    ("sorento_2024", "Kia Sorento 2024 exterior"),
    ("sorento_2025", "Kia Sorento 2025 exterior"),
    ("sorento_2026", "Kia Sorento 2026 exterior"),

    # --------------------------------------------------------
    # GENERACIONES
    # --------------------------------------------------------

    ("primera_generacion", "Kia Sorento first generation exterior"),
    ("segunda_generacion", "Kia Sorento second generation exterior"),
    ("tercera_generacion", "Kia Sorento third generation exterior"),
    ("cuarta_generacion", "Kia Sorento fourth generation exterior"),

    ("sorento_bl", "Kia Sorento BL exterior"),
    ("sorento_xm", "Kia Sorento XM exterior"),
    ("sorento_um", "Kia Sorento UM exterior"),
    ("sorento_mq4", "Kia Sorento MQ4 exterior"),

    # --------------------------------------------------------
    # VERSIONES Y EQUIPAMIENTOS
    # --------------------------------------------------------

    ("sorento_ls", "Kia Sorento LS exterior"),
    ("sorento_lx", "Kia Sorento LX exterior"),
    ("sorento_ex", "Kia Sorento EX exterior"),
    ("sorento_sx", "Kia Sorento SX exterior"),
    ("sorento_sx_prestige", "Kia Sorento SX Prestige exterior"),
    ("sorento_gt_line", "Kia Sorento GT Line exterior"),
    ("sorento_x_line", "Kia Sorento X-Line exterior"),
    ("sorento_x_pro", "Kia Sorento X-Pro exterior"),
    ("sorento_touring", "Kia Sorento Touring exterior"),
    ("sorento_limited", "Kia Sorento Limited exterior"),

    # --------------------------------------------------------
    # MOTORIZACIONES Y VERSIONES ESPECIALES
    # --------------------------------------------------------

    ("sorento_diesel", "Kia Sorento diesel exterior"),
    ("sorento_gasolina", "Kia Sorento gasoline exterior"),
    ("sorento_hybrid", "Kia Sorento Hybrid exterior"),
    ("sorento_phev", "Kia Sorento Plug in Hybrid exterior"),
    ("sorento_v6", "Kia Sorento V6 exterior"),
    ("sorento_4x4", "Kia Sorento 4x4 exterior"),
    ("sorento_awd", "Kia Sorento AWD exterior"),

    # --------------------------------------------------------
    # PERSPECTIVAS
    # --------------------------------------------------------

    ("frontal", "Kia Sorento front view exterior"),
    ("trasera", "Kia Sorento rear view exterior"),
    ("lateral", "Kia Sorento side view exterior"),
    ("tres_cuartos_frontal", "Kia Sorento front three quarter view"),
    ("tres_cuartos_trasero", "Kia Sorento rear three quarter view"),
    ("frontal_diagonal", "Kia Sorento front diagonal exterior photograph"),
    ("trasera_diagonal", "Kia Sorento rear diagonal exterior photograph"),

    # --------------------------------------------------------
    # ESCENARIOS
    # --------------------------------------------------------

    ("parqueadero", "Kia Sorento parked in parking lot"),
    ("calle", "Kia Sorento driving on city street"),
    ("concesionario", "Kia Sorento exterior car dealership"),
    ("carretera", "Kia Sorento driving on highway"),
    ("montana", "Kia Sorento exterior mountain road"),
    ("campo", "Kia Sorento exterior countryside"),
    ("luz_dia", "Kia Sorento exterior daylight photography"),
    ("noche", "Kia Sorento exterior at night"),
    ("lluvia", "Kia Sorento exterior rainy weather"),
    ("frente_edificio", "Kia Sorento parked outside building"),
    ("varios_vehiculos", "Kia Sorento parked among other cars"),

    # --------------------------------------------------------
    # COLOMBIA
    # --------------------------------------------------------

    ("sorento_colombia", "Kia Sorento Colombia fotografías exterior"),
    ("sorento_usado", "Kia Sorento usado Colombia exterior"),
    ("sorento_2010_colombia", "Kia Sorento 2010 Colombia exterior"),
    ("sorento_2015_colombia", "Kia Sorento 2015 Colombia exterior"),
    ("sorento_2020_colombia", "Kia Sorento 2020 Colombia exterior"),
    ("sorento_2024_colombia", "Kia Sorento 2024 Colombia exterior"),
    ("placa_colombiana", "Kia Sorento con placa colombiana exterior"),

]


# ============================================================
# PARÁMETROS DE DESCARGA
# ============================================================

ANCHO_MINIMO = 640
ALTO_MINIMO = 420

BYTES_MINIMOS = 15 * 1024
BYTES_MAXIMOS = 25 * 1024 * 1024

DISTANCIA_DUPLICADOS = 5

TIMEOUT = 25
REINTENTOS = 3


# ============================================================
# UTILIDADES
# ============================================================

def limpiar_nombre(texto):

    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9_-]+", "_", texto)

    return texto.strip("_")


def sha256_bytes(contenido):

    return hashlib.sha256(contenido).hexdigest()


def hash_perceptual(imagen):

    gris = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    reducido = cv2.resize(
        gris,
        (8, 8),
        interpolation=cv2.INTER_AREA
    )

    promedio = reducido.mean()
    bits = reducido > promedio

    valor = 0

    for bit in bits.flatten():
        valor = (valor << 1) | int(bit)

    return valor


def distancia_hash(hash1, hash2):

    return bin(hash1 ^ hash2).count("1")


# ============================================================
# CARGAR IMÁGENES EXISTENTES
# ============================================================

def cargar_hashes_existentes():

    hashes_sha = set()
    hashes_visual = []

    for archivo in CARPETA_SALIDA.iterdir():

        if archivo.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".webp"
        ]:
            continue

        try:

            contenido = archivo.read_bytes()

            imagen = cv2.imdecode(
                np.frombuffer(contenido, np.uint8),
                cv2.IMREAD_COLOR
            )

            if imagen is None:
                continue

            hashes_sha.add(
                sha256_bytes(contenido)
            )

            hashes_visual.append(
                hash_perceptual(imagen)
            )

        except Exception as error:

            print(
                f"[AVISO] No se pudo leer "
                f"{archivo.name}: {error}"
            )

    if METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as archivo:

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

def parametros_busqueda(consulta, pagina):

    return {
        "q": consulta,
        "gl": "co",
        "hl": "es",
        "page": pagina,
    }


def buscar(consulta, pagina, api_key):

    config = PROVEEDORES[PROVEEDOR_ACTIVO]

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    parametros = parametros_busqueda(
        consulta,
        pagina
    )

    for intento in range(REINTENTOS):

        try:

            respuesta = requests.post(
                config["url"],
                headers=headers,
                json=parametros,
                timeout=TIMEOUT
            )

            if respuesta.status_code == 429:

                print(
                    "[AVISO] Límite de solicitudes alcanzado."
                )

                time.sleep(5 * (intento + 1))
                continue

            if respuesta.status_code in [401, 403]:

                raise RuntimeError(
                    "API Key inválida o sin permisos en Serper.dev."
                )

            respuesta.raise_for_status()

            datos = respuesta.json()

            return datos.get("images", [])

        except requests.exceptions.Timeout:

            print("[AVISO] Tiempo de espera agotado.")

        except requests.exceptions.RequestException as error:

            print(f"[ERROR] Error de búsqueda: {error}")

        if intento < REINTENTOS - 1:
            time.sleep(2)

    return []


def imagen_de_resultado(resultado):

    return (
        resultado.get("imageUrl")
        or resultado.get("thumbnailUrl")
    )


def fuente_de_resultado(resultado):

    fuente = resultado.get("source", "")

    if isinstance(fuente, dict):

        return (
            fuente.get("name")
            or fuente.get("link")
            or ""
        )

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
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        respuesta.raise_for_status()

        contenido = respuesta.content

        if not (
            BYTES_MINIMOS
            <= len(contenido)
            <= BYTES_MAXIMOS
        ):
            return None

        imagen = cv2.imdecode(
            np.frombuffer(contenido, np.uint8),
            cv2.IMREAD_COLOR
        )

        if imagen is None:
            return None

        alto, ancho = imagen.shape[:2]

        if (
            ancho < ANCHO_MINIMO
            or alto < ALTO_MINIMO
        ):
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

        if distancia_hash(
            hash_visual,
            existente
        ) <= DISTANCIA_DUPLICADOS:

            return False

    nombre_base = limpiar_nombre(consulta_id)

    nombre_archivo = (
        f"sorento_{nombre_base}_{siguiente_numero:05d}.jpg"
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
        "modelo": "kia_sorento",
        "consulta": consulta_id,
        "url_imagen": imagen_de_resultado(resultado),
        "url_origen": resultado.get("link", ""),
        "fuente": fuente_de_resultado(resultado),
        "ancho": int(imagen.shape[1]),
        "alto": int(imagen.shape[0]),
        "sha256": hash_sha,
        "fecha_descarga": datetime.now().isoformat(),

    }

    with open(
        METADATA_FILE,
        "a",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            json.dumps(
                metadato,
                ensure_ascii=False
            ) + "\n"
        )

    return True


# ============================================================
# INTERCALAR RESULTADOS
# ============================================================

def intercalar_consultas(candidatos):

    resultado = []

    maximo = max(
        (len(lista) for lista in candidatos.values()),
        default=0
    )

    for posicion in range(maximo):

        for consulta_id, lista in candidatos.items():

            if posicion < len(lista):

                resultado.append(
                    (consulta_id, lista[posicion])
                )

    return resultado


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def ejecutar(args):

    api_key = (
        args.api_key
        or os.getenv(
            PROVEEDORES[PROVEEDOR_ACTIVO]["clave"]
        )
    )

    if api_key:
        api_key = api_key.strip()

    if not api_key:

        raise RuntimeError(
            "Falta la API Key. Utiliza --api-key "
            "o configura SERPER_API_KEY."
        )

    hashes_sha, hashes_visual = cargar_hashes_existentes()

    print("\n======================================")
    print("       DESCARGADOR KIA SORENTO")
    print("======================================")

    print(f"Imágenes existentes: {len(hashes_sha)}")
    print(f"Objetivo acumulado: {args.total}")
    print(f"Consultas: {len(CONSULTAS)}")
    print(f"Páginas por consulta: {args.paginas}")

    if len(hashes_sha) >= args.total:

        print("\nYa se alcanzó el objetivo de imágenes.")
        return

    candidatos = {}

    # --------------------------------------------------------
    # FASE 1: BUSCAR IMÁGENES
    # --------------------------------------------------------

    for consulta_id, consulta in CONSULTAS:

        candidatos[consulta_id] = []

        print(f"\n[BUSCANDO] {consulta}")

        for pagina in range(
            args.pagina_inicial,
            args.pagina_inicial + args.paginas
        ):

            resultados = buscar(
                consulta,
                pagina,
                api_key
            )

            print(
                f"   Página {pagina}: "
                f"{len(resultados)} resultados"
            )

            for resultado in resultados:

                if imagen_de_resultado(resultado):

                    candidatos[consulta_id].append(
                        resultado
                    )

            time.sleep(args.pausa)

    # --------------------------------------------------------
    # FASE 2: INTERCALAR RESULTADOS
    # --------------------------------------------------------

    lista_final = intercalar_consultas(candidatos)

    print("\n======================================")
    print("COMENZANDO DESCARGA")
    print("======================================")

    descargadas = 0
    siguiente_numero = len(hashes_sha) + 1

    # --------------------------------------------------------
    # FASE 3: DESCARGAR Y VALIDAR
    # --------------------------------------------------------

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
                f"Total: {len(hashes_sha)} | "
                f"{consulta_id}"
            )

        else:

            print(
                f"[DUPLICADA] {consulta_id}"
            )

        time.sleep(args.pausa)

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    print("\n======================================")
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
        description="Descargador de imágenes Kia Sorento con Serper.dev"
    )

    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Cantidad total objetivo de imágenes"
    )

    parser.add_argument(
        "--paginas",
        type=int,
        default=1,
        help="Páginas por cada consulta"
    )

    parser.add_argument(
        "--pagina-inicial",
        type=int,
        default=1,
        help="Número de la primera página"
    )

    parser.add_argument(
        "--pausa",
        type=float,
        default=1.0,
        help="Pausa entre solicitudes"
    )

    parser.add_argument(
        "--ancho-min",
        type=int,
        default=640,
        help="Ancho mínimo de imagen"
    )

    parser.add_argument(
        "--alto-min",
        type=int,
        default=420,
        help="Alto mínimo de imagen"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key de Serper.dev"
    )

    args = parser.parse_args()

    global ANCHO_MINIMO
    global ALTO_MINIMO

    ANCHO_MINIMO = args.ancho_min
    ALTO_MINIMO = args.alto_min

    if args.total < 1 or args.paginas < 1:
        parser.error("--total y --paginas deben ser mayores que cero")

    ejecutar(args)


if __name__ == "__main__":
    main()