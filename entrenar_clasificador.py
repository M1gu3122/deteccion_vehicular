import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un clasificador de marca/modelo y lo exporta a OpenVINO."
    )
    parser.add_argument(
        "--datos",
        default=str(BASE_DIR / "dataset_autos"),
        help="Carpeta con train/ y val/, una subcarpeta por clase.",
    )
    parser.add_argument(
        "--modelo",
        default="yolo11n-cls.pt",
        help="Modelo base de clasificacion de Ultralytics.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument(
        "--nombre",
        default="auto_cls",
        help="Nombre del modelo resultante (modelos/<nombre>.pt).",
    )
    args = parser.parse_args()

    datos = Path(args.datos)

    if not (datos / "train").is_dir():
        print(f"Falta la carpeta de entrenamiento: {datos / 'train'}")
        print("Estructura esperada:")
        print(f"  {datos}\\train\\kia_picanto\\*.jpg")
        print(f"  {datos}\\train\\kia_rio\\*.jpg")
        print(f"  {datos}\\val\\kia_picanto\\*.jpg")
        return

    modelo = YOLO(args.modelo)
    modelo.train(
        data=str(datos),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(BASE_DIR / "runs"),
        name=args.nombre,
        exist_ok=True,
    )

    mejor = Path(modelo.trainer.best)
    MODELOS_DIR.mkdir(exist_ok=True)
    destino_pt = MODELOS_DIR / f"{args.nombre}.pt"
    shutil.copy2(mejor, destino_pt)
    print(f"Modelo guardado en: {destino_pt}")

    exportado = YOLO(str(destino_pt)).export(
        format="openvino",
        imgsz=args.imgsz,
    )

    carpeta_exportada = Path(exportado)
    destino_ov = MODELOS_DIR / f"{args.nombre}_openvino_model"

    if carpeta_exportada.resolve() != destino_ov.resolve():
        if destino_ov.exists():
            shutil.rmtree(destino_ov)
        shutil.move(str(carpeta_exportada), str(destino_ov))

    print(f"OpenVINO listo en: {destino_ov}")


if __name__ == "__main__":
    main()
