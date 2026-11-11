"""
Reconoce marca/modelo de un vehículo en una imagen.
Usa modelos/auto_cls_v2.pt para clasificación.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# ═══════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
MODELO_CLASIFICACION = BASE_DIR / "modelos" / "auto_cls_v2.pt"

CLASES_KIA = [
    "kia_amanti", "kia_forte", "kia_optima", "kia_picanto",
    "kia_rio", "kia_sedona", "kia_sephia", "kia_sorento",
    "kia_soul", "kia_spectra", "kia_sportage"
]

# Cargar modelo UNA sola vez
try:
    clasificador = YOLO(str(MODELO_CLASIFICACION), task="classify")
except Exception as e:
    print(f"❌ No se pudo cargar el modelo: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════

def predecir_marca_modelo(imagen, conf_min=0.45):
    """Clasifica una imagen con el modelo de marca/modelo."""
    try:
        resultado = clasificador.predict(imagen, conf=conf_min, imgsz=224, verbose=False)[0]
        if resultado.probs is None:
            return None
        idx = int(resultado.probs.top1)
        confianza = float(resultado.probs.top1conf)
        if confianza < conf_min:
            return None
        return CLASES_KIA[idx], confianza
    except Exception as e:
        print(f"⚠️ Error en clasificación: {e}")
        return None

# ══════════════════════════════════════════════
# INTERFAZ DE LÍNEA DE COMPAÑA
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Clasifica la marca y modelo de un vehículo en una imagen.")
    parser.add_argument("--imagen", required=True, help="Ruta de la imagen a evaluar.")
    parser.add_argument("--directo", action="store_true", help="Clasifica la imagen entera.")
    parser.add_argument("--confianza", type=float, default=0.45, help="Confianza mínima.")
    args = parser.parse_args()

    ruta_imagen = Path(args.imagen)
    if not ruta_imagen.is_absolute():
        ruta_imagen = Path.cwd() / ruta_imagen

    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        print(f"❌ No se pudo cargar: {ruta_imagen}")
        sys.exit(1)

    print(f"\n🔍 Evaluando: {ruta_imagen.name}")
    print("=" * 50)

    resultado = predecir_marca_modelo(imagen, conf_min=args.confianza)

    if resultado is None:
        print("❌ No se pudo clasificar la imagen (puede ser muy pequeña o oscura)")
    else:
        etiqueta, conf = resultado
        print(f"\n✅ Marca/modelo: {etiqueta} ({conf*100:.1f}% de confianza)")

    # Guardar resultado sobre la imagen
    output_path = Path("resultados") / "clasificacion_resultado.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.putText(
        imagen,
        f"Auto: {etiqueta} {conf:.2f}" if resultado else "",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output_path), imagen)
    print(f"\n💾 Resultado guardado en: {output_path}")

    if not args.directo:
        # Mostrar ventana solo si no es modo directo (o siempre, es simple)
        try:
            cv2.imshow("Resultado", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()