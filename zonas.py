import json
from pathlib import Path


ARCHIVO_ZONA = Path(__file__).resolve().parent / "zona_ingreso.json"


def guardar_zona(puntos, ancho, alto):
    datos = {
        "nombre": "Zona de ingreso",
        "ancho_video": ancho,
        "alto_video": alto,
        "puntos": puntos
    }

    with open(ARCHIVO_ZONA, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=4, ensure_ascii=False)

    print(f"Zona guardada en: {ARCHIVO_ZONA}")


def cargar_zona():
    if not ARCHIVO_ZONA.exists():
        return None

    with open(ARCHIVO_ZONA, "r", encoding="utf-8") as archivo:
        return json.load(archivo)