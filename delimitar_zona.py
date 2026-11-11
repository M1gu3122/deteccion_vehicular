
import cv2
import numpy as np
from pathlib import Path

from zonas import guardar_zona, cargar_zona

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "videos" / "entrada.webm"

puntos = []
fotograma_original = None
fotograma_seleccion = None


def dibujar_zona():
    global fotograma_seleccion

    fotograma_seleccion = fotograma_original.copy()

    for i, punto in enumerate(puntos):
        cv2.circle(fotograma_seleccion, punto, 6, (0, 255, 255), -1)
        cv2.putText(
            fotograma_seleccion,
            str(i + 1),
            (punto[0] + 8, punto[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    if len(puntos) >= 2:
        for i in range(len(puntos) - 1):
            cv2.line(
                fotograma_seleccion,
                puntos[i],
                puntos[i + 1],
                (0, 255, 0),
                2
            )

    if len(puntos) == 4:
        poligono = np.array(puntos, dtype=np.int32)
        capa = fotograma_seleccion.copy()

        cv2.fillPoly(capa, [poligono], (0, 255, 0))
        fotograma_seleccion = cv2.addWeighted(
            capa, 0.20, fotograma_seleccion, 0.80, 0
        )

        cv2.polylines(
            fotograma_seleccion, [poligono], True, (0, 255, 0), 2
        )

    cv2.imshow("Delimitar zona", fotograma_seleccion)


def evento_raton(event, x, y, flags, parametro):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
        puntos.append((x, y))
        dibujar_zona()


def reproducir_video():
    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        print("No se pudo abrir el video.")
        return

    datos = cargar_zona()
    if not datos:
        print("No existe una zona guardada.")
        video.release()
        return

    poligono = np.array(datos["puntos"], dtype=np.int32)
    ancho_zona = datos["ancho_video"]
    alto_zona = datos["alto_video"]

    ancho_video = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_video = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (ancho_video, alto_video) != (ancho_zona, alto_zona):
        print("La resolución del video no coincide con la zona guardada.")
        print("Vuelve a delimitar la zona con este video.")
        video.release()
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    retraso = max(1, int(1000 / fps)) if fps > 0 else 33

    cv2.namedWindow("Video con zona", cv2.WINDOW_NORMAL)

    while True:
        correcto, fotograma = video.read()
        if not correcto:
            break

        capa = fotograma.copy()
        cv2.fillPoly(capa, [poligono], (0, 255, 0))
        fotograma = cv2.addWeighted(capa, 0.20, fotograma, 0.80, 0)

        cv2.polylines(fotograma, [poligono], True, (0, 255, 0), 3)
        cv2.putText(
            fotograma,
            "Zona delimitada",
            tuple(poligono[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Video con zona", fotograma)

        tecla = cv2.waitKey(retraso) & 0xFF
        if tecla == 27 or tecla == ord("q"):
            break

        # Barra espaciadora: pausar o reanudar.
        if tecla == ord(" "):
            while True:
                tecla_pausa = cv2.waitKey(0) & 0xFF
                if tecla_pausa == ord(" ") or tecla_pausa == 27:
                    break
            if tecla_pausa == 27:
                break

    video.release()
    cv2.destroyAllWindows()


def main():
    global fotograma_original

    if not VIDEO_PATH.exists():
        print(f"No se encontró el video: {VIDEO_PATH}")
        return

    video = cv2.VideoCapture(str(VIDEO_PATH))
    correcto, fotograma = video.read()
    video.release()

    if not correcto:
        print("No se pudo leer el video.")
        return

    fotograma_original = fotograma.copy()
    alto, ancho = fotograma_original.shape[:2]

    cv2.namedWindow("Delimitar zona", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Delimitar zona", evento_raton)
    dibujar_zona()

    print("Haz clic en cuatro puntos.")
    print("S: guardar zona | R: reiniciar | P: reproducir | ESC: salir")

    while True:
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord("s"):
            if len(puntos) != 4:
                print("Selecciona cuatro puntos antes de guardar.")
                continue

            guardar_zona(
                [list(punto) for punto in puntos], ancho, alto
            )
            print("Zona guardada.")

        elif tecla == ord("r"):
            puntos.clear()
            dibujar_zona()

        elif tecla == ord("p"):
            if len(puntos) == 4:
                guardar_zona(
                    [list(punto) for punto in puntos], ancho, alto
                )
                cv2.destroyWindow("Delimitar zona")
                reproducir_video()
                return
            else:
                print("Primero selecciona cuatro puntos.")

        elif tecla == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
