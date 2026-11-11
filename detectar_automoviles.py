
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from zonas import cargar_zona

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "salida.webm"

# Detector local: clase 1 = vehiculo.
MODELO = BASE_DIR / "modelos" / "best.pt"

# Detector local: clase 1 = vehiculo.
CLASE_AUTOMOVIL = 1
CONFIANZA_MINIMA = 0.40


def punto_dentro_zona(punto, poligono):
    return cv2.pointPolygonTest(
        poligono,
        (float(punto[0]), float(punto[1])),
        False
    ) >= 0


def main():
    datos_zona = cargar_zona()

    if not datos_zona:
        print("No hay una zona guardada. Ejecuta primero delimitar_zona.py.")
        return

    poligono = np.array(datos_zona["puntos"], dtype=np.int32)

    if not MODELO.is_file():
        raise FileNotFoundError(f"Falta el detector local: {MODELO}")
    modelo = YOLO(str(MODELO))
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print(f"No se pudo abrir el video: {VIDEO_PATH}")
        return

    ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (ancho, alto) != (
        datos_zona["ancho_video"],
        datos_zona["alto_video"]
    ):
        print("La resolución del video no coincide con la zona guardada.")
        video.release()
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    retraso = max(1, int(1000 / fps)) if fps > 0 else 33

    cv2.namedWindow("Deteccion de automoviles", cv2.WINDOW_NORMAL)

    try:
        while True:
            correcto, frame = video.read()

            if not correcto:
                break

            # Detectar únicamente la clase automóvil.
            resultados = modelo.predict(
                source=frame,
                classes=[CLASE_AUTOMOVIL],
                conf=CONFIANZA_MINIMA,
                imgsz=640,
                verbose=False
            )

            # Dibujar la zona.
            capa = frame.copy()
            cv2.fillPoly(capa, [poligono], (0, 255, 0))
            frame = cv2.addWeighted(capa, 0.15, frame, 0.85, 0)
            cv2.polylines(frame, [poligono], True, (0, 255, 0), 2)

            automoviles_dentro = 0
            automoviles_fuera = 0

            for resultado in resultados:
                if resultado.boxes is None:
                    continue

                for caja in resultado.boxes:
                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0].tolist()
                    )

                    confianza = float(caja.conf[0])

                    # Punto inferior central del automóvil.
                    # Es más útil para zonas de suelo que el centro
                    # del recuadro, especialmente con cámaras elevadas.
                    punto = ((x1 + x2) // 2, y2)

                    dentro = punto_dentro_zona(punto, poligono)

                    if dentro:
                        automoviles_dentro += 1
                        color = (0, 255, 0)
                        estado = "DENTRO"
                    else:
                        automoviles_fuera += 1
                        color = (0, 165, 255)
                        estado = "FUERA"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )

                    etiqueta = f"Automovil {confianza:.2f} | {estado}"

                    cv2.putText(
                        frame,
                        etiqueta,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    cv2.circle(frame, punto, 4, color, -1)

            cv2.putText(
                frame,
                f"Dentro: {automoviles_dentro} | Fuera: {automoviles_fuera}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Deteccion de automoviles", frame)

            tecla = cv2.waitKey(retraso) & 0xFF

            if tecla == ord("q") or tecla == 27:
                break

            if tecla == ord(" "):
                while True:
                    tecla_pausa = cv2.waitKey(0) & 0xFF
                    if tecla_pausa == ord(" ") or tecla_pausa == 27:
                        break
                if tecla_pausa == 27:
                    break

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
