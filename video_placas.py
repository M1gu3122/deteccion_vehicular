import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from zonas import cargar_zona
from lector_ocr import LectorOCR
from clasificador_vehiculo import ClasificadorVehiculo


BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "videos" / "2017 KIA RIO.mp4"
MODELO = BASE_DIR / "modelos" / "best.pt"

# Indices de clase del modelo (names: {0: '0' -> placa, 1: '2' -> vehiculo})
CLASE_PLACA = 0
CLASE_VEHICULO = 1

CONFIANZA_INFERENCIA = 0.30
CONFIANZA_VEHICULO = 0.40
MARGEN_PLACA = 0.08

# Tamano de inferencia (coincide con el export de OpenVINO)
TAMANO_INFERENCIA = 416

# Reintentar el OCR de una placa sin lectura cada N fotogramas
OCR_REINTENTO_FOTOGRAMAS = 20


def cargar_modelo(ruta):
    """Carga la version OpenVINO si existe; si no, el .pt original."""
    ruta = Path(ruta)
    carpeta_ov = ruta.parent / f"{ruta.stem}_openvino_model"
    if carpeta_ov.exists():
        return YOLO(str(carpeta_ov), task="detect")
    return YOLO(str(ruta))


def punto_dentro_zona(punto, poligono):
    return cv2.pointPolygonTest(
        poligono, (float(punto[0]), float(punto[1])), False
    ) >= 0


def recortar_placa(frame, coordenadas, margen=MARGEN_PLACA):
    x1, y1, x2, y2 = coordenadas
    alto, ancho = frame.shape[:2]

    margen_x = int((x2 - x1) * margen)
    margen_y = int((y2 - y1) * margen)

    x1 = max(0, x1 - margen_x)
    y1 = max(0, y1 - margen_y)
    x2 = min(ancho, x2 + margen_x)
    y2 = min(alto, y2 + margen_y)

    return frame[y1:y2, x1:x2]


def placa_plausible(coordenadas, ancho_frame):
    x1, y1, x2, y2 = coordenadas
    ancho = x2 - x1
    alto = y2 - y1
    if alto < 8 or ancho < 20:
        return False
    # Descartar cajas enormes (vehiculo completo) o demasiado alargadas
    if ancho > 0.6 * ancho_frame:
        return False
    relacion = ancho / max(1, alto)
    return 1.2 <= relacion <= 8.0


def clave_placa(coordenadas, id_track):
    """Identifica una placa entre fotogramas para cachear su lectura."""
    if id_track is not None:
        return int(id_track)
    x1, y1, x2, y2 = coordenadas
    return ((x1 + x2) // 40, (y1 + y2) // 40)


def dibujar_zona(frame, poligono):
    """Dibuja la zona solo sobre su region para no copiar el frame entero."""
    x, y, ancho, alto = cv2.boundingRect(poligono)
    if ancho <= 0 or alto <= 0:
        return

    roi = frame[y:y + alto, x:x + ancho]
    capa = roi.copy()
    desplazado = poligono - np.array([x, y], dtype=np.int32)
    cv2.fillPoly(capa, [desplazado], (0, 255, 0))
    frame[y:y + alto, x:x + ancho] = cv2.addWeighted(capa, 0.15, roi, 0.85, 0)
    cv2.polylines(frame, [poligono], True, (0, 255, 0), 2)


def main():
    datos_zona = cargar_zona()

    if not datos_zona:
        print("No existe una zona guardada. Ejecuta delimitar_zona.py.")
        return

    if not MODELO.exists():
        print(f"No se encontro el modelo: {MODELO}")
        return

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print(f"No se pudo abrir el video: {VIDEO_PATH}")
        return

    ancho_video = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_video = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     if (ancho_video, alto_video) != (
#         datos_zona["ancho_video"],
#         datos_zona["alto_video"],
# ):
#         print("El video no coincide con la resolucion de la zona guardada.")
#         video.release()
#         return

    poligono = np.array(datos_zona["puntos"], dtype=np.int32)

    try:
        modelo = cargar_modelo(MODELO)
        lector_ocr = LectorOCR()
        clasificador = ClasificadorVehiculo()
    except Exception as error:
        video.release()
        print(f"No se pudieron cargar el modelo o el OCR: {error}")
        return

    fps_video = video.get(cv2.CAP_PROP_FPS)
    espera_base = 1000.0 / fps_video if fps_video > 0 else 33.0

    cv2.namedWindow("Automoviles y reconocimiento de placas", cv2.WINDOW_NORMAL)

    lecturas = {}
    ultimo_intento = {}
    marcas = {}
    indice = 0
    t_anterior = time.time()
    fps_mostrado = 0.0

    try:
        while True:
            t_frame = time.time()
            correcto, frame = video.read()

            if not correcto:
                break

            # Una sola pasada del modelo: placas y vehiculos a la vez.
            resultado = modelo.track(
                frame,
                conf=CONFIANZA_INFERENCIA,
                imgsz=TAMANO_INFERENCIA,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            ids = resultado.boxes.id if resultado.boxes is not None else None

            vehiculos = []
            placas = []

            for posicion, caja in enumerate(resultado.boxes or []):
                clase = int(caja.cls[0])
                confianza = float(caja.conf[0])
                x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())

                if clase == CLASE_VEHICULO and confianza >= CONFIANZA_VEHICULO:
                    id_track = int(ids[posicion]) if ids is not None else None
                    vehiculos.append((x1, y1, x2, y2, confianza, id_track))

                elif clase == CLASE_PLACA and placa_plausible(
                    (x1, y1, x2, y2), ancho_video
                ):
                    id_track = int(ids[posicion]) if ids is not None else None
                    placas.append((x1, y1, x2, y2, confianza, id_track))

            # OCR de placas sobre el frame limpio (antes de dibujar nada).
            textos_placa = []
            for x1, y1, x2, y2, confianza_placa, id_track in placas:
                clave = clave_placa((x1, y1, x2, y2), id_track)
                texto = lecturas.get(clave, "")
                necesita_ocr = clave not in lecturas or (
                    not texto
                    and indice - ultimo_intento.get(clave, -999)
                    >= OCR_REINTENTO_FOTOGRAMAS
                )

                if necesita_ocr:
                    recorte = recortar_placa(frame, (x1, y1, x2, y2))
                    if recorte.size > 0:
                        ultimo_intento[clave] = indice
                        try:
                            texto = lector_ocr.leer(recorte)["placa_normalizada"]
                            lecturas[clave] = texto
                        except Exception as error:
                            print(f"Error OCR: {error}")

                textos_placa.append(lecturas.get(clave, ""))

            # Clasificar marca/modelo una vez por vehiculo (sobre frame limpio).
            if clasificador.disponible:
                for x1, y1, x2, y2, _, id_track in vehiculos:
                    clave = ("auto", clave_placa((x1, y1, x2, y2), id_track))
                    if clave in marcas:
                        continue
                    try:
                        prediccion = clasificador.predecir(
                            frame[y1:y2, x1:x2]
                        )
                    except Exception as error:
                        print(f"Error clasificador: {error}")
                        prediccion = None
                    if prediccion is not None:
                        marcas[clave] = prediccion

            # Dibujar zona.
            dibujar_zona(frame, poligono)

            # Dibujar vehiculos y contar dentro/fuera.
            autos_dentro = 0
            autos_fuera = 0
            for x1, y1, x2, y2, confianza, id_track in vehiculos:
                punto = ((x1 + x2) // 2, y2)

                if punto_dentro_zona(punto, poligono):
                    autos_dentro += 1
                    color, estado = (0, 255, 0), "DENTRO"
                else:
                    autos_fuera += 1
                    color, estado = (0, 165, 255), "FUERA"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"Automovil {confianza:.2f} | {estado}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                cv2.circle(frame, punto, 4, color, -1)

                clave = ("auto", clave_placa((x1, y1, x2, y2), id_track))
                marca = marcas.get(clave)
                if marca is not None:
                    etiqueta_marca, confianza_marca = marca
                    cv2.putText(
                        frame,
                        f"{etiqueta_marca} {confianza_marca:.2f}",
                        (x1, min(alto_video - 8, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

            # Dibujar placas con su lectura.
            for (x1, y1, x2, y2, confianza_placa, _), texto in zip(
                placas, textos_placa
            ):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                etiqueta = texto or "SIN LECTURA"
                cv2.putText(
                    frame,
                    f"{etiqueta} ({confianza_placa:.2f})",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2,
                )

            # FPS suavizado.
            ahora = time.time()
            fps_instantaneo = 1.0 / max(1e-6, ahora - t_anterior)
            fps_mostrado = (
                fps_instantaneo
                if fps_mostrado == 0
                else 0.9 * fps_mostrado + 0.1 * fps_instantaneo
            )
            t_anterior = ahora

            cv2.putText(
                frame,
                f"Automoviles dentro: {autos_dentro} | fuera: {autos_fuera}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps_mostrado:.1f}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Automoviles y reconocimiento de placas", frame)

            # Esperar solo el tiempo restante para reproducir a velocidad real.
            transcurrido = (time.time() - t_frame) * 1000.0
            espera = int(max(1, espera_base - transcurrido))
            tecla = cv2.waitKey(espera) & 0xFF

            if tecla in (ord("q"), 27):
                break

            if tecla == ord(" "):
                while True:
                    tecla_pausa = cv2.waitKey(0) & 0xFF
                    if tecla_pausa in (ord(" "), 27, ord("q")):
                        break
                if tecla_pausa in (27, ord("q")):
                    break

            indice += 1

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
