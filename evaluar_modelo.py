"""
Evalua el clasificador Kia y detecta si el modelo colapso a una sola clase.

Sintoma del entrenamiento anterior: el modelo devolvia "picanto" con 99.9% de
confianza para cualquier imagen. Este script comprueba clase por clase si el
modelo realmente aprendio a distinguir los modelos.

Uso:
    python evaluar_modelo.py
    python evaluar_modelo.py --modelo modelos/auto_cls_v2.pt
"""

import argparse
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_autos_v2"
MODELOS_DIR = BASE_DIR / "modelos"

CLASES = ['amanti', 'forte', 'optima', 'picanto', 'rio', 'sedona',
          'sephia', 'sorento', 'soul', 'spectra', 'sportage']


def cargar_modelo(ruta):
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="classify")
    return YOLO(str(ruta), task="classify")


def main():
    parser = argparse.ArgumentParser(description="Evalua el clasificador Kia.")
    parser.add_argument("--modelo", default=str(MODELOS_DIR / "auto_cls_v2.pt"))
    parser.add_argument("--datos", default=str(DATASET_DIR))
    parser.add_argument("--confianza", type=float, default=0.45)
    args = parser.parse_args()

    datos = Path(args.datos)
    ruta_modelo = Path(args.modelo)

    if not ruta_modelo.exists():
        ruta_modelo = MODELOS_DIR / "auto_cls.pt"
    if not ruta_modelo.exists():
        print(f"No se encontro el modelo: {args.modelo}")
        return

    modelo = cargar_modelo(ruta_modelo)
    nombres = modelo.names
    print("=" * 74)
    print(f"EVALUACION DEL MODELO: {ruta_modelo.name}")
    print("=" * 74)
    print(f"Clases: {dict(sorted(nombres.items(), key=lambda x: x[1]))}\n")

    # ------------------------------------------------- predicciones por clase
    print("=" * 74)
    print("PREDICCION CLASE POR CLASE (conjunto de validacion)")
    print("=" * 74)

    aciertos_por_clase = defaultdict(int)
    totales_por_clase = defaultdict(int)
    predicciones = defaultdict(int)
    matriz = defaultdict(lambda: defaultdict(int))
    confianza_promedio = defaultdict(list)

    for clase in CLASES:
        carpeta = datos / "val" / f"kia_{clase}"
        if not carpeta.is_dir():
            continue

        imagenes = sorted(carpeta.glob("*.jpg"))
        if not imagenes:
            continue

        resultados = modelo.predict(
            [str(p) for p in imagenes],
            imgsz=224,
            verbose=False,
        )

        for resultado in resultados:
            real = clase
            if resultado.probs is None:
                continue
            indice = int(resultado.probs.top1)
            predicho = nombres[indice]
            confianza = float(resultado.probs.top1conf)

            totales_por_clase[real] += 1
            predicciones[predicho] += 1
            matriz[real][predicho] += 1
            confianza_promedio[real].append(confianza)
            if predicho == real:
                aciertos_por_clase[real] += 1

    print(f"\n{'clase':<12} {'n':>4} {'aciertos':>9} {'recall':>8} "
          f"{'conf.media':>11} {'conf.min':>9}")
    print("-" * 74)

    total_n = total_ok = 0
    for clase in CLASES:
        n = totales_por_clase.get(clase, 0)
        if n == 0:
            continue
        ok = aciertos_por_clase.get(clase, 0)
        recall = ok / n
        confs = confianza_promedio.get(clase, [0])
        print(f"{clase:<12} {n:>4} {ok:>9} {recall:>7.1%} "
              f"{sum(confs) / len(confs):>10.1%} {min(confs):>8.1%}")
        total_n += n
        total_ok += ok

    print("-" * 74)
    print(f"{'GLOBAL':<12} {total_n:>4} {total_ok:>9} "
          f"{total_ok / total_n if total_n else 0:>7.1%}")

    # ------------------------------------------------- distribucion
    print("\n" + "=" * 74)
    print("DISTRIBUCION DE PREDICCIONES (detecta colapso a una clase)")
    print("=" * 74)
    print(f"{'clase predicha':<18} {'veces':>7} {'%':>8}")
    print("-" * 74)
    for clase, n in sorted(predicciones.items(), key=lambda x: -x[1]):
        print(f"{clase:<18} {n:>7} {n / total_n if total_n else 0:>7.1%}")

    maximo = max(predicciones.values()) if predicciones else 0
    if total_n and maximo / total_n > 0.70:
        dominante = max(predicciones, key=predicciones.get)
        print(f"\n  [ALERTA] El modelo predice '{dominante}' en el "
              f"{maximo / total_n:.0%} de los casos.")
        print("  Esto es el mismo fallo del entrenamiento anterior: "
              "el modelo no aprendio a distinguir los modelos.")
    else:
        print("\n  [OK] No hay colapso hacia una sola clase.")

    # ------------------------------------------------- confusion matrix
    print("\n" + "=" * 74)
    print("MATRIZ DE CONFUSION (fila = real, columna = predicho)")
    print("=" * 74)
    print(f"{'real\\pred':<12}" + "".join(f"{c[:7]:>8}" for c in CLASES))
    for real in CLASES:
        if totales_por_clase.get(real, 0) == 0:
            continue
        fila = "".join(
            f"{matriz[real].get(p, 0):>8}" for p in CLASES
        )
        print(f"{real:<12}{fila}")

    # ------------------------------------------------- top confusiones
    print("\n" + "=" * 74)
    print("CONFUSIONES MAS FRECUENTES")
    print("=" * 74)
    pares = []
    for real in CLASES:
        for predicho, n in matriz[real].items():
            if real != predicho and n > 0:
                pares.append((n, real, predicho))
    pares.sort(reverse=True)
    for n, real, predicho in pares[:12]:
        print(f"  {n:>3}x  {real:<12} -> {predicho}")
    if not pares:
        print("  Sin confusiones: todas las predicciones fueron correctas.")

    print("\n" + "=" * 74)
    print("INTERPRETACION")
    print("=" * 74)
    if total_n and maximo / total_n > 0.70:
        print("  El modelo colapso. Revisa: dataset desbalanceado, recortes")
        print("  de calidad desigual o etiquetas incorrectas en el dataset.")
    elif total_n and total_ok / total_n >= 0.85:
        print("  Buen resultado. El modelo distingue los modelos correctamente.")
        print("  Verifica con imagenes reales antes de deployarlo.")
    elif total_n:
        print("  Resultado parcial. Revisa las confusiones de arriba:")
        print("  puede hacer falta mas augmentation o mas imagenes por clase.")
    print("=" * 74)


if __name__ == "__main__":
    main()
