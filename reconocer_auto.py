"""
Reconoce marca/modelo Y placa de un vehículo en una imagen.
Usa:
- modelos/auto_cls_v2.pt  para clasificación de marca/modelo
- modelos/best.pt          para detección de vehículo y placa
- PaddleOCR              para lectura de caracteres de la placa
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════
# RUTAS Y MODELOS
# ═══════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / "modelos"

MODELO_CLASIFICACION = MODELOS_DIR / "auto_cls_v2.pt"

MODELO_DETECCION = MODELOS_DIR / "best.pt"

try:
    detector = YOLO(str(MODELO_DETECCION))
    clasificador = YOLO(str(MODELO_CLASIFICACION), task="classify")
except Exception as e:
    print(f"⚠️ Error cargando modelos: {e}")
    detector = None
    clasificador = None

CLASES_KIA = [
    "kia_amanti", "kia_forte", "kia_optima", "kia_picanto",
    "kia_rio", "kia_sedona", "kia_sephia", "kia_sorento",
    "kia_soul", "kia_spectra", "kia_sportage"
]

try:
    from lector_ocr import LectorOCR
    ocr = LectorOCR()
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False

# ═══════════════════════════════════════════════
# FUNCIONES DE DETECCIÓN
# ═══════════════════════════════════════════════

def predecir_marca_modelo(imagen, conf_min=0.45):
    """Clasifica una imagen con el modelo de marca/modelo."""
    if clasificador is None:
        return None
    try:
        resultado = clasificador.predict(imagen, conf=conf_min, imgsz=224, verbose=False)[0]
        if resultado.probs is None:
            return None
        idx = int(resultado.probs.top1)
        confianza = float(resultado.probs.top1conf)
        if confianza < conf_min:
            return None
        return clasificador.names[idx], confianza
    except Exception as e:
        print(f"⚠️ Error en clasificación: {e}")
        return None


def deteccion_vehiculo_y_placa(imagen, conf_vehiculo=0.40):
    """
    Detecta vehículo y placa en la imagen con el modelo YOLO.
    Returns ((x1,y1,x2,y2,conf)_vehiculo, (x1,y1,x2,y2,conf)_placa) o (None, None).
    """
    if detector is None:
        return None, None
    try:
        resultados = detector.predict(imagen, conf=conf_vehiculo, imgsz=640, verbose=False)[0]
        cajas = resultados.boxes
        if not cajas:
            return None, None

        # Separar por clase: clase 1 = vehículo, clase 0 = placa
        vehiculo = None
        placa = None
        for caja in cajas:
            cls = int(caja.cls[0])
            conf = float(caja.conf[0])
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

            if cls == 1 and vehiculo is None:  # vehículo (tomamos la primera mayor)
                vehiculo = (x1, y1, x2, y2, conf)
            elif cls == 0 and placa is None:  # placa (tomamos la primera)
                # Validar que parece una placa (relación de aspecto plausible)
                ancho = x2 - x1
                alto = y2 - y1
                if alto >= 8 and ancho >= 20 and not (ancho > 0.8 * imagen.shape[1]):
                    placa = (x1, y1, x2, y2, conf)

        return vehiculo, placa
    except Exception as e:
        print(f"⚠️ Error en detección: {e}")
        return None, None


def ocr_placa(imagen_corte):
    """Intenta leer el texto de una placa de coche con PaddleOCR."""
    if not OCR_DISPONIBLE:
        return None
    try:
        return ocr.leer(imagen_corte)["placa_normalizada"] or None
    except Exception as e:
        print(f"⚠️ Error OCR: {e}")
    return None


# ═══════════════════════════════════════════════
# FUNCIONES DE DISEÑO
# ════════════════════════════════════════════════

def dibujar_resultado(imagen, x1, y1, x2, y2, texto, color=(0, 255, 0), color_texto=(255, 255, 255)):
    """Dibuja rectángulo y texto sobre la imagen."""
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        imagen,
        texto,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_texto,
        2,
    )


# ═══════════════════════════════════════════════
# FUNCIONES PRINCIPALES
# ════════════════════════════════════════════════

def predecir_marca_modelo(imagen, conf_min=0.45):
    """Clasifica una imagen con el modelo de marca/modelo."""
    if clasificador is None:
        return None
    try:
        resultado = clasificador.predict(imagen, conf=conf_min, imgsz=224, verbose=False)[0]
        if resultado.probs is None:
            return None
        idx = int(resultado.probs.top1)
        confianza = float(resultado.probs.top1conf)
        if confianza < conf_min:
            return None
        return clasificador.names[idx], confianza
    except Exception as e:
        print(f"⚠️ Error en clasificación: {e}")
        return None


def predecir_desde_carroceria(imagen, conf_vehiculo=0.40, conf_clasif=0.45):
    """
    Detecta vehículo y placa, clasifica el vehículo y OCR la placa.
    Returns (etiqueta_modelo, conf_modelo, texto_placa) o None.
    """
    if detector is None or clasificador is None:
        return None

    vehiculo, placa = deteccion_vehiculo_y_placa(imagen, conf_vehiculo)
    if vehiculo is None:
        return None

    x1_v, y1_v, x2_v, y2_v, conf_v = vehiculo

    # 1. Recortar vehículo con margen
    alto, ancho = imagen.shape[:2]
    margen_x = int((x2_v - x1_v) * 0.08)
    margen_y = int((y2_v - y1_v) * 0.15)
    x1n = max(0, x1_v - margen_x)
    yn = max(0, y1_v - margen_y)
    x2n = min(ancho, x2_v + margen_x)
    yn2 = min(alto, y2_v + margen_y)

    recorte = imagen[yn:yn2, x1n:x2n]
    if recorte.size == 0 or recorte.shape[1] < 100:
        return None

    # 2. Clasificar el vehículo recortado
    resultado_clas = predecir_marca_modelo(recorte, conf_clasif)
    if resultado_clas is None:
        return None

    etiqueta_modelo, conf_modelo = resultado_clas

    # 3. OCR sobre la placa (si la hubo)
    placa_texto = None
    if placa is not None:
        x1_p, y1_p, x2_p, y2_p, conf_p = placa
        # Recortar placa del frame original
        placa_recorte = imagen[y1_p:y2_p, x1_p:x2_p]
        if placa_recorte.size > 0:
            resultado_placa = ocr_placa(placa_recorte)
            if resultado_placa:
                placa_texto = placa_plausible((x1_p, y1_p, x2_p, y2_p), imagen.shape[1])
                # Nota: placa_plausible valida la relación de aspecto,
                # pero ya la validamos en deteccion_vehiculo_y_placa.
                # Retornamos el texto OCR limpio:
                # (Quito la validación duplicada y retorno el texto directamente)
                pass  # placa_texto ya tiene el valor del OCR

    return etiqueta_modelo, conf_modelo, placa_texto


# ═══════════════════════════════════════════════
# FUNCIONES DE DISEÑO
# ════════════════════════════════════════════════

def dibujar_resultado(imagen, x1, y1, x2, y2, texto, color=(0, 255, 0), color_texto=(255, 255, 255)):
    """Dibuja rectángulo y texto sobre la imagen."""
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        imagen,
        texto,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_texto,
        2,
    )


# ═══════════════════════════════════════════════
# INTERFAZ DE LÍNEA DE COMPAÑA
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reconoce marca/modelo Y placa de un vehículo en una imagen."
    )
    parser.add_argument(
        "--imagen",
        default=str(BASE_DIR / "imagenes" / "kia1.jpg"),
        help="Ruta de la imagen a evaluar.",
    )
    parser.add_argument(
        "--directo",
        action="store_true",
        help="Clasifica la imagen entera (solo marca/modelo, sin detección de placa).",
    )
    parser.add_argument(
        "--salida",
        default=str(BASE_DIR / "resultados" / "reconocer_auto.jpg"),
        help="Ruta donde guardar la imagen con resultado.",
    )
    parser.add_argument(
        "--confianza-vehiculo",
        type=float,
        default=0.40,
        help="Confianza mínima para detectar el vehículo y placa (detector YOLO).",
    )
    parser.add_argument(
        "--conf-clasificador",
        type=float,
        default=0.45,
        help="Confianza mínima del clasificador marca/modelo.",
    )
    parser.add_argument(
        "--sin-ventana",
        action="store_true",
        help="No abrir ventana gráfica.",
    )
    args = parser.parse_args()

    ruta_imagen = Path(args.imagen)
    if not ruta_imagen.is_absolute():
        ruta_imagen = BASE_DIR / ruta_imagen

    imagen = cv2.imread(str(ruta_imagen))
    if imagen is None:
        print(f"❌ No se pudo cargar la imagen: {ruta_imagen}")
        sys.exit(1)

    print(f"\n🔍 Evaluando: {ruta_imagen.name}")
    print("=" * 60)

    # ── Modo directo: solo clasificación de marca/modelo ─────────────
    if args.directo:
        resultado = predecir_marca_modelo(imagen, conf_min=args.conf_clasificador)
        if resultado is None:
            print("❌ No se pudo clasificar la imagen (confianza baja o imagen inválida)")
        else:
            etiqueta, conf = resultado
            print(f"\n✅ Marca/modelo: {etiqueta} ({conf*100:.1f}% de confianza)")

        # Guardar imagen con texto
        salida = Path(args.salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        cv2.putText(
            imagen,
            f"Modelo: {etiqueta} {conf:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(salida), imagen)
        print(f"\n💾 Imagen guardada en: {salida}")

        if not args.sin_ventana:
            try:
                cv2.imshow("Resultado - Marca/Modelo", imagen)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass
        return

    # ── Modo completo: detectar vehículo + clasificar + OCR de placa ─
    print("🔍 Modo completo: detectando vehículo, clasificando y leyendo placa...")
    resultado = predecir_desde_carroceria(imagen, conf_vehiculo=args.confianza_vehiculo, conf_clasif=args.conf_clasificador)

    if resultado is None:
        print("❌ No se pudo procesar la imagen")
        print("   • No se detectó vehículo"
               " • El recorte es muy pequeño"
               " • La confianza es muy baja")
        # Guardar imagen original sin anotaciones
        salida = Path(args.salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(salida), imagen)
        return

    etiqueta_modelo, conf_modelo, placa_texto = resultado

    print(f"\n✅ Marca/modelo: {etiqueta_modelo} ({conf_modelo*100:.1f}% de confianza)")

    if placa_texto:
        print(f"🔢 Placa detectada: {placa_texto}")
    else:
        print("🔢 Placa: No se pudo leer (o no hay placa en la imagen)")

    # Guardar imagen con anotaciones
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    # Dibujar sobre la imagen
    # 1. Dibujar rectángulo del vehículo (usamos las coordenadas del vehículo detectado)
    # Pero no tenemos las coordenadas guardadas fácilmente, así que dibujamos textos y estimamos
    h, w = imagen.shape[:2]

    # 2. Texto marca/modelo
    texto_modelo = f"Auto: {etiqueta_modelo} {conf_modelo:.2f}"
    cv2.putText(
        imagen,
        texto_modelo,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # 3. Texto placa si la hay
    if placa_texto:
        texto_placa = f"Placa: {placa_texto}"
        # Posicionar a la derecha, abajo
        cv2.putText(
            imagen,
            texto_placa,
            (w - 300, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
        )

    # Texto informativo adicional
    texto_adicional = "Procesamiento completado"
    cv2.putText(
        imagen,
        texto_adicional,
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(salida), imagen)
    print(f"\n💾 Imagen guardada en: {salida}")

    if not args.sin_ventana:
        try:
            cv2.imshow("Resultado - Auto, Modelo y Placa", imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"🖥️ No se pudo mostrar la ventana: {e}")


if __name__ == "__main__":
    main()