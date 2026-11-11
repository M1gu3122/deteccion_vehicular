"""
Entrena el clasificador de marca/modelo Kia sobre el dataset reconstruido.

Uso:
    python entrenar_clasificador_v2.py
    python entrenar_clasificador_v2.py --epochs 120 --imgsz 256
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_autos_v2"
MODELOS_DIR = BASE_DIR / "modelos"
RUNS_DIR = BASE_DIR / "runs"


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el clasificador Kia con parametros corregidos."
    )
    parser.add_argument("--modelo", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "False"])
    parser.add_argument("--nombre", default="auto_cls_v2")
    parser.add_argument("--solo-comando", action="store_true",
                        help="Muestra el comando sin ejecutarlo.")
    args = parser.parse_args()

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"Falta {data_yaml}")
        print("Ejecuta primero:  python preparar_dataset_v2.py")
        return

    argumentos = {
        "data": str(DATASET_DIR),  # Carpeta dataset_autos_v2/ (directorio, no archivo yaml)
        "model": args.modelo,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": True,
        "project": str(RUNS_DIR),
        "name": args.nombre,
        "exist_ok": True,
        "seed": 0,
        "deterministic": True,
        "cache": args.cache,
        "workers": 4,
        "close_mosaic": args.close_mosaic,
        # Data augmentation corregido para evitar que el modelo aprenda
        # "imagen limpia = picanto"
        "degrees": 12.0,
        "translate": 0.15,
        "scale": 0.45,
        "shear": 2.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 1.0,
        "mixup": 0.25,
        "copy_paste": 0.05,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.25,
        "dropout": 0.1,
    }

    print("=" * 74)
    print("ENTRENAMIENTO CLASIFICADOR KIA v2")
    print("=" * 74)
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Modelo:   {args.modelo}")
    print(f"Epochs:   {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print(f"Optimizer:{args.optimizer}  lr0: {args.lr0}  patience: {args.patience}")
    print("=" * 74)
    print("\nComando equivalente:")
    print(f"  yolo classify train data={DATASET_DIR} model={args.modelo} "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"patience={args.patience} optimizer={args.optimizer}")

    if args.solo_comando:
        return

    print("\n" + "=" * 74)
    print("INICIANDO ENTRENAMIENTO (puede tardar bastante)")
    print("=" * 74 + "\n")

    inicio = time.time()
    modelo = YOLO(args.modelo)
    modelo.train(**argumentos)

    # --------------------------------------------- guardar el mejor modelo
    mejor = Path(modelo.trainer.best)
    MODELOS_DIR.mkdir(exist_ok=True)
    destino_pt = MODELOS_DIR / f"{args.nombre}.pt"
    shutil.copy2(mejor, destino_pt)
    print(f"\nModelo guardado en: {destino_pt}")

    # --------------------------------------------- exportar a OpenVINO
    print("\nExportando a OpenVINO...")
    exportado = YOLO(str(destino_pt)).export(format="openvino", imgsz=args.imgsz)
    carpeta_exportada = Path(exportado)
    destino_ov = MODELOS_DIR / f"{args.nombre}_openvino_model"

    if carpeta_exportada.resolve() != destino_ov.resolve():
        if destino_ov.exists():
            shutil.rmtree(destino_ov)
        shutil.move(str(carpeta_exportada), str(destino_ov))

    print(f"OpenVINO listo en: {destino_ov}")

    # --------------------------------------------- activar el modelo nuevo
    clasificador = BASE_DIR / "clasificador_vehiculo.py"
    if clasificador.exists():
        texto = clasificador.read_text(encoding="utf-8")
        if f'MODELO_CLASIFICADOR = MODELOS_DIR / "{args.nombre}.pt"' not in texto:
            texto = texto.replace(
                'MODELO_CLASIFICADOR = MODELOS_DIR / "auto_cls.pt"',
                f'MODELO_CLASIFICADOR = MODELOS_DIR / "{args.nombre}.pt"'
            )
            clasificador.write_text(texto, encoding="utf-8")
            print(f"clasificador_vehiculo.py ahora usa {args.nombre}.pt")

    fin = time.time()
    print(f"\nTiempo total: {(fin - inicio)/60:.1f} minutos")

    print("\n" + "=" * 74)
    print("LISTO. Ahora evalua el modelo:")
    print("  python evaluar_modelo.py")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())