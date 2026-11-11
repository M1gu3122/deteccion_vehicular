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

CARPETA_SALIDA = BASE_DIR / "dataset_autos" / "kia_spectra"
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
# CONSULTAS: KIA SPECTRA EXTERIOR
# ============================================================

CONSULTAS = [

    # AÑOS
    ("spectra_2000", "Kia Spectra 2000 exterior"),
    ("spectra_2001", "Kia Spectra 2001 exterior"),
    ("spectra_2002", "Kia Spectra 2002 exterior"),
    ("spectra_2003", "Kia Spectra 2003 exterior"),
    ("spectra_2004", "Kia Spectra 2004 exterior Colombia"),
    ("spectra_2005", "Kia Spectra 2005 exterior"),
    ("spectra_2006", "Kia Spectra 2006 exterior"),
    ("spectra_2007", "Kia Spectra 2007 exterior"),
    ("spectra_2008", "Kia Spectra 2008 exterior"),
    ("spectra_2009", "Kia Spectra 2009 exterior"),

    # GENERACIONES
    ("primera_generacion", "Kia Spectra first generation exterior"),
    ("segunda_generacion", "Kia Spectra second generation exterior"),
    ("spectra_2000_2003", "Kia Spectra 2000 2001 2002 2003 exterior"),
    ("spectra_2004_2009", "Kia Spectra 2004 2005 2006 2007 2008 2009 exterior"),

    # CARROCERÍAS
    ("spectra_sedan", "Kia Spectra sedan exterior"),
    ("spectra_hatchback", "Kia Spectra hatchback exterior"),
    ("spectra5", "Kia Spectra5 hatchback exterior"),
    ("spectra_ls", "Kia Spectra LS sedan exterior"),
    ("spectra_lx", "Kia Spectra LX sedan exterior"),
    ("spectra_ex", "Kia Spectra EX sedan exterior"),
    ("spectra_sx", "Kia Spectra SX sedan exterior"),

    # VERSIONES SPECTRA5
    ("spectra5_sx", "Kia Spectra5 SX exterior"),
    ("spectra5_2005", "Kia Spectra5 2005 exterior"),
    ("spectra5_2006", "Kia Spectra5 2006 exterior"),
    ("spectra5_2007", "Kia Spectra5 2007 exterior"),
    ("spectra5_2008", "Kia Spectra5 2008 exterior"),
    ("spectra5_2009", "Kia Spectra5 2009 exterior"),

    # PERSPECTIVAS
    ("frontal", "Kia Spectra front view exterior"),
    ("trasera", "Kia Spectra rear view exterior"),
    ("lateral", "Kia Spectra side view exterior"),
    ("tres_cuartos_frontal", "Kia Spectra front three quarter view"),
    ("tres_cuartos_trasero", "Kia Spectra rear three quarter view"),

    # ESCENARIOS
    ("parqueadero", "Kia Spectra parked in parking lot exterior"),
    ("calle", "Kia Spectra driving on city street"),
    ("concesionario", "Kia Spectra exterior car dealership"),
    ("luz_dia", "Kia Spectra exterior daylight photography"),
    ("noche", "Kia Spectra exterior at night"),
    ("lluvia", "Kia Spectra exterior rainy weather"),
    ("calle_colombia", "Kia Spectra parked on Colombian street"),
    ("placa_colombiana", "Kia Spectra Colombian license plate exterior"),

    # COLOMBIA
    ("spectra_colombia", "Kia Spectra Colombia fotografías exterior"),
    ("spectra_ls_colombia", "Kia Spectra LS Colombia exterior"),
    ("spectra_2004_colombia", "Kia Spectra 2004 Colombia exterior"),
    ("spectra_usado", "Kia Spectra usado Colombia fotografías exterior"),
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

            print(f"[AVISO] No se pudo leer {archivo.name}: {error}")

    # Recuperar hashes registrados en metadatos
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

                print("[AVISO] Límite de solicitudes alcanzado.")
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
        f"spectra_{nombre_base}_{siguiente_numero:05d}.jpg"
    )

    ruta = CARPETA_SALIDA / nombre_archivo

    exito, codificado = cv2.imencode(
        ".jpg",
        imagen,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not exito:
        return False

    contenido_final = codificado.tobytes()

    ruta.write_bytes(contenido_final)

    hashes_sha.add(hash_sha)
    hashes_visual.append(hash_visual)

    metadato = {

        "archivo": nombre_archivo,
        "modelo": "kia_spectra",
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

    # Prioridad: API Key recibida por argumento.
    # Alternativa: variable de entorno.

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
    print("       DESCARGADOR KIA SPECTRA")
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
        description="Descargador de imágenes Kia Spectra con Serper.dev"
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

    ejecutar(args)


if __name__ == "__main__":
    main()