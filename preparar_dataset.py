import argparse
import hashlib
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELO = BASE_DIR / "modelos" / "best.pt"

CLASE_VEHICULO = 1
CONFIANZA = 0.40
TAMANO_INFERENCIA = 416
MARGEN = 0.05

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_modelo(ruta):
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="detect")
    return YOLO(str(ruta))


def listar_imagenes(carpeta):
    return sorted(
        ruta
        for ruta in carpeta.rglob("*")
        if ruta.suffix.lower() in EXTENSIONES
    )


def recortar_vehiculos(modelo, imagen, confianza, min_ancho):
    alto, ancho = imagen.shape[:2]

    resultado = modelo.predict(
        imagen,
        classes=[CLASE_VEHICULO],
        conf=confianza,
        imgsz=TAMANO_INFERENCIA,
        verbose=False,
    )[0]

    recortes = []

    for caja in resultado.boxes or []:
        x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

        margen_x = int((x2 - x1) * MARGEN)
        margen_y = int((y2 - y1) * MARGEN)

        x1 = max(0, x1 - margen_x)
        y1 = max(0, y1 - margen_y)
        x2 = min(ancho, x2 + margen_x)
        y2 = min(alto, y2 + margen_y)

        if x2 - x1 < min_ancho:
            continue

        recorte = imagen[y1:y2, x1:x2]

        if recorte.size > 0:
            recortes.append(recorte)

    return recortes


def main():
    parser = argparse.ArgumentParser(
        description="Organiza imagenes propias en train/val por clase."
    )
    parser.add_argument(
        "--entrada",
        default=str(BASE_DIR / "crudo"),
        help="Carpeta con una subcarpeta por clase (kia_picanto, kia_rio...).",
    )
    parser.add_argument(
        "--salida",
        default=str(BASE_DIR / "dataset_autos"),
        help="Carpeta del dataset resultante (train/ y val/).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.20,
        help="Fraccion para validacion (0.20 = 20%%).",
    )
    parser.add_argument(
        "--recortar",
        action="store_true",
        help="Recorta el vehiculo con best.pt antes de guardar.",
    )
    parser.add_argument("--confianza", type=float, default=CONFIANZA)
    parser.add_argument("--min-ancho", type=int, default=80)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    entrada = Path(args.entrada)

    if not entrada.is_dir():
        print(f"No existe la carpeta: {entrada}")
        print("Crea una subcarpeta por clase, por ejemplo:")
        print(f"  {entrada}\\kia_picanto\\*.jpg")
        print(f"  {entrada}\\kia_rio\\*.jpg")
        return

    clases = sorted(ruta for ruta in entrada.iterdir() if ruta.is_dir())

    if not clases:
        print(f"No hay subcarpetas de clase dentro de {entrada}")
        return

    modelo = None

    if args.recortar:
        if not MODELO.exists():
            print(f"No se encontro el detector: {MODELO}")
            return
        modelo = cargar_modelo(MODELO)

    random.seed(args.semilla)
    salida = Path(args.salida)

    for clase_dir in clases:
        clase = clase_dir.name
        candidatos = []
        vistas = set()

        for ruta in listar_imagenes(clase_dir):
            imagen = cv2.imread(str(ruta))

            if imagen is None:
                print(f"  {clase}: no se pudo leer {ruta.name}")
                continue

            if modelo is not None:
                recortes = recortar_vehiculos(
                    modelo, imagen, args.confianza, args.min_ancho
                )
            else:
                recortes = [imagen]

            for recorte in recortes:
                correcto, buffer = cv2.imencode(".jpg", recorte)

                if not correcto:
                    continue

                datos = buffer.tobytes()
                firma = hashlib.sha1(datos).hexdigest()[:16]

                if firma in vistas:
                    continue

                vistas.add(firma)
                candidatos.append((firma, datos))

        if not candidatos:
            print(f"{clase}: sin imagenes utiles")
            continue

        random.shuffle(candidatos)
        total = len(candidatos)

        if total < 2:
            n_val = total
            aviso = " (muy pocas imagenes: se duplica en val)"
        else:
            n_val = max(1, int(round(total * args.val)))
            n_val = min(n_val, total - 1)
            aviso = ""

        for posicion, (firma, datos) in enumerate(candidatos):
            particiones = ["val"] if posicion < n_val else ["train"]

            if total < 2:
                particiones = ["train", "val"]

            for particion in particiones:
                carpeta = salida / particion / clase
                carpeta.mkdir(parents=True, exist_ok=True)
                destino = carpeta / f"{clase}_{firma}.jpg"
                destino.write_bytes(datos)

        print(
            f"{clase}: {total} imagenes "
            f"(train={total - n_val}, val={n_val}){aviso}"
        )

    print(f"\nDataset listo en: {salida}")
    print(
        "Siguiente paso: python entrenar_clasificador.py "
        "--datos dataset_autos --epochs 80"
    )


if __name__ == "__main__":
    main()
