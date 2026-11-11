"""Analiza imagenes o videos Kia: modelo, color estimado y lectura de placa."""

import argparse
import json
from pathlib import Path

import cv2

from clasificador_vehiculo import ClasificadorVehiculo
from color_vehiculo import estimar_color
from config import MODELO_PLACAS
from detector import cargar_modelo


IMAGENES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def asociar_placas(vehiculos, placas):
    """Asigna cada placa al vehiculo contenedor mas pequeno, una sola vez."""
    grupos = [[] for _ in vehiculos]
    for placa in placas:
        px1, py1, px2, py2 = placa["caja"]
        candidatos = []
        for indice, auto in enumerate(vehiculos):
            x1, y1, x2, y2 = auto["caja"]
            if x1 <= px1 and y1 <= py1 and x2 >= px2 and y2 >= py2:
                candidatos.append(((x2 - x1) * (y2 - y1), indice))
        if candidatos:
            grupos[min(candidatos)[1]].append(placa)
    return grupos


class AnalizadorVehiculo:
    def __init__(self, sin_ocr=False, confianza=.45):
        if not MODELO_PLACAS.is_file():
            raise FileNotFoundError(f"Falta el detector: {MODELO_PLACAS}")
        self.detector = cargar_modelo(MODELO_PLACAS)
        self.clasificador = ClasificadorVehiculo(confianza_minima=confianza)
        if not self.clasificador.disponible:
            raise RuntimeError("No se pudo cargar tu clasificador de 11 modelos Kia.")
        self.lector = None
        if not sin_ocr:
            from lector_ocr import LectorOCR
            self.lector = LectorOCR()

    def analizar(self, imagen):
        alto, ancho = imagen.shape[:2]
        prediccion = self.detector.predict(imagen, conf=.30, imgsz=416, verbose=False)[0]
        vehiculos, placas = [], []
        if prediccion.boxes is not None:
            for caja in prediccion.boxes:
                x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
                coordenadas = [max(0, x1), max(0, y1), min(ancho, x2), min(alto, y2)]
                if coordenadas[2] <= coordenadas[0] or coordenadas[3] <= coordenadas[1]:
                    continue
                clase, confianza = int(caja.cls[0]), float(caja.conf[0])
                dato = {"caja": coordenadas, "confianza": round(confianza, 4)}
                if clase == 1 and confianza >= .40:
                    vehiculos.append(dato)
                elif clase == 0:
                    placas.append(dato)
        grupos = asociar_placas(vehiculos, placas)
        for auto, placas_auto in zip(vehiculos, grupos):
            x1, y1, x2, y2 = auto["caja"]
            recorte = imagen[y1:y2, x1:x2]
            resultado = self.clasificador.predecir(recorte)
            etiqueta, confianza = resultado if resultado else (None, None)
            auto.update(marca="Kia", modelo=etiqueta.removeprefix("kia_") if etiqueta else "indeterminado",
                        confianza_modelo=round(confianza, 4) if confianza is not None else None)
            auto["color"] = estimar_color(recorte, [
                [p["caja"][0] - x1, p["caja"][1] - y1,
                 p["caja"][2] - x1, p["caja"][3] - y1] for p in placas_auto])
            auto["placas"] = placas_auto
        # Leer tambien placas sin vehiculo asociado, manteniendolas separadas.
        for placa in placas:
            x1, y1, x2, y2 = placa["caja"]
            dx, dy = max(2, int((x2 - x1) * .08)), max(2, int((y2 - y1) * .15))
            recorte = imagen[max(0, y1-dy):min(alto, y2+dy),
                             max(0, x1-dx):min(ancho, x2+dx)]
            placa.update(texto="", estado="ocr_desactivado" if self.lector is None else "sin_lectura")
            if self.lector is not None:
                try:
                    lectura = self.lector.leer(recorte)
                    placa.update(texto=lectura["placa_normalizada"],
                                 texto_original=lectura["texto_original"],
                                 confianzas_ocr=[float(c) for c in lectura["confianzas_ocr"]])
                    if placa["texto"]:
                        placa["estado"] = "leida"
                except Exception as error:
                    placa.update(estado="error_ocr", error=str(error))
        asignadas = {id(p) for grupo in grupos for p in grupo}
        return {"vehiculos": vehiculos,
                "placas_sin_vehiculo": [p for p in placas if id(p) not in asignadas]}


def dibujar(imagen, resultado):
    salida = imagen.copy()
    placas = list(resultado["placas_sin_vehiculo"])
    for auto in resultado["vehiculos"]:
        x1, y1, x2, y2 = auto["caja"]
        cv2.rectangle(salida, (x1, y1), (x2, y2), (0, 220, 0), 2)
        texto = f"Kia {auto['modelo']} | {auto['color']['nombre']}"
        cv2.putText(salida, texto, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 0), 2)
        placas.extend(auto["placas"])
    for placa in placas:
        x1, y1, x2, y2 = placa["caja"]
        cv2.rectangle(salida, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(salida, placa["texto"] or "SIN LECTURA", (x1, max(20, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 255), 2)
    return salida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrada", type=Path, help="Ruta a una foto o un video.")
    parser.add_argument("--salida", type=Path, default=Path("resultados/analisis"), help="Carpeta nueva de resultados.")
    parser.add_argument("--mostrar", action="store_true", help="Mostrar una ventana; Q termina el video.")
    parser.add_argument("--sin-ocr", action="store_true", help="Detectar placas sin leer texto (no inicializa PaddleOCR).")
    parser.add_argument("--cada", type=int, default=1, help="Analizar cada N fotogramas; los omitidos se guardan sin anotaciones.")
    parser.add_argument("--confianza", type=float, default=.45, help="Umbral de clasificacion del modelo Kia.")
    args = parser.parse_args()
    if not args.entrada.is_file():
        parser.error(f"No existe la entrada: {args.entrada}")
    if args.cada < 1 or not 0 <= args.confianza <= 1:
        parser.error("--cada debe ser positivo y --confianza debe estar entre 0 y 1.")
    extension = args.entrada.suffix.lower()
    if extension not in IMAGENES | VIDEOS:
        parser.error("Formato no admitido; usa una imagen o un video.")
    imagen = None
    captura = None
    if extension in IMAGENES:
        imagen = cv2.imread(str(args.entrada))
        if imagen is None:
            parser.error("No se pudo decodificar la imagen.")
    else:
        captura = cv2.VideoCapture(str(args.entrada))
        if not captura.isOpened():
            captura.release()
            parser.error("No se pudo abrir el video.")
    escritor = None
    try:
        if args.salida.exists():
            parser.error("La carpeta de salida ya existe; elige otra con --salida para conservar los resultados.")
        analizador = AnalizadorVehiculo(args.sin_ocr, args.confianza)
        args.salida.mkdir(parents=True)
        if imagen is not None:
            resultado = analizador.analizar(imagen)
            salida = dibujar(imagen, resultado)
            if not cv2.imwrite(str(args.salida / "imagen.jpg"), salida):
                raise RuntimeError("No se pudo guardar la imagen.")
            (args.salida / "datos.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(resultado, ensure_ascii=False, indent=2))
            if args.mostrar:
                cv2.imshow("Analisis Kia", salida)
                cv2.waitKey(0)
        else:
            fps = captura.get(cv2.CAP_PROP_FPS)
            fps = fps if 0 < fps < 1000 else 25.0
            indice = 0
            with (args.salida / "datos.jsonl").open("w", encoding="utf-8") as datos:
                while True:
                    correcto, frame = captura.read()
                    if not correcto:
                        break
                    if escritor is None:
                        alto, ancho = frame.shape[:2]
                        escritor = cv2.VideoWriter(str(args.salida / "video.mp4"),
                                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
                        if not escritor.isOpened():
                            raise RuntimeError("No se pudo crear el video MP4.")
                    if indice % args.cada == 0:
                        resultado = analizador.analizar(frame)
                        datos.write(json.dumps({"fotograma": indice, "segundos": round(indice/fps, 3),
                                                **resultado}, ensure_ascii=False) + "\n")
                        datos.flush()
                        frame = dibujar(frame, resultado)
                        print(f"Fotograma {indice}: {len(resultado['vehiculos'])} vehiculos")
                    escritor.write(frame)
                    indice += 1
                    if args.mostrar:
                        cv2.imshow("Analisis Kia", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            if indice == 0:
                raise RuntimeError("El video no contiene fotogramas decodificables.")
        print(f"Resultados guardados en: {args.salida.resolve()}")
    finally:
        if captura is not None:
            captura.release()
        if escritor is not None:
            escritor.release()
        if args.mostrar:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
