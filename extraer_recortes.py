import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELO = BASE_DIR / "modelos" / "best.pt"

# Indice de clase del vehiculo en el modelo (names: {0: placa, 1: vehiculo})
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


def archivos_entrada(ruta):
    ruta = Path(ruta)
    if ruta.is_dir():
        return sorted(
            p for p in ruta.rglob("*") if p.suffix.lower() in EXTENSIONES
        )
    return [ruta]


def main():
    parser = argparse.ArgumentParser(
        description="Extrae recortes de vehiculos para armar un dataset de marca/modelo."
    )
    parser.add_argument(
        "--entrada",
        default=str(BASE_DIR / "imagenes"),
        help="Imagen o carpeta de imagenes de entrada.",
    )
    parser.add_argument(
        "--salida",
        default=str(BASE_DIR / "dataset_autos" / "_recortes"),
        help="Carpeta donde guardar los recortes.",
    )
    parser.add_argument("--min-ancho", type=int, default=80)
    args = parser.parse_args()

    if not MODELO.exists():
        print(f"No se encontro el modelo: {MODELO}")
        return

    modelo = cargar_modelo(MODELO)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    total = 0

    for ruta in archivos_entrada(args.entrada):
        imagen = cv2.imread(str(ruta))

        if imagen is None:
            print(f"No se pudo leer: {ruta}")
            continue

        alto, ancho = imagen.shape[:2]

        resultado = modelo.predict(
            imagen,
            classes=[CLASE_VEHICULO],
            conf=CONFIANZA,
            imgsz=TAMANO_INFERENCIA,
            verbose=False,
        )[0]

        for indice, caja in enumerate(resultado.boxes or []):
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

            margen_x = int((x2 - x1) * MARGEN)
            margen_y = int((y2 - y1) * MARGEN)

            x1 = max(0, x1 - margen_x)
            y1 = max(0, y1 - margen_y)
            x2 = min(ancho, x2 + margen_x)
            y2 = min(alto, y2 + margen_y)

            if x2 - x1 < args.min_ancho:
                continue

            recorte = imagen[y1:y2, x1:x2]

            if recorte.size == 0:
                continue

            nombre = f"{ruta.stem}_{indice:03d}.jpg"
            cv2.imwrite(str(salida / nombre), recorte)
            total += 1

    print(f"Recortes guardados: {total} en {salida}")


if __name__ == "__main__":
    main()
