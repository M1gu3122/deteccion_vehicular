import ctypes

import cv2


def _tamano_pantalla():
    try:
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


class VisorImagen:
    """Visor 1:1 con zoom y arrastre para imagenes de cualquier tamano.

    Inicia ajustado a la pantalla; con la rueda se acerca al 100% real y
    arrastrando con el boton izquierdo se recorre la imagen.
    """

    FACTOR_ZOOM = 1.15

    def __init__(self, nombre, imagen):
        self.nombre = nombre
        self.imagen = imagen
        self.alto, self.ancho = imagen.shape[:2]

        pantalla_ancho, pantalla_alto = _tamano_pantalla()
        max_ancho = int(pantalla_ancho * 0.90)
        max_alto = int((pantalla_alto - 80) * 0.90)

        self.escala_min = min(
            1.0,
            max_ancho / self.ancho,
            max_alto / self.alto,
        )
        self.escala_max = 4.0
        self.escala = self.escala_min

        self.ancho_vista = max(1, int(self.ancho * self.escala_min))
        self.alto_vista = max(1, int(self.alto * self.escala_min))

        self.cx = self.ancho / 2.0
        self.cy = self.alto / 2.0

        self._arrastrando = False
        self._ultimo = (0, 0)

    def _region(self):
        ancho_region = min(self.ancho, self.ancho_vista / self.escala)
        alto_region = min(self.alto, self.alto_vista / self.escala)

        mitad_ancho = ancho_region / 2.0
        mitad_alto = alto_region / 2.0

        self.cx = min(max(self.cx, mitad_ancho), self.ancho - mitad_ancho)
        self.cy = min(max(self.cy, mitad_alto), self.alto - mitad_alto)

        x1 = int(round(self.cx - mitad_ancho))
        y1 = int(round(self.cy - mitad_alto))
        x2 = int(round(x1 + ancho_region))
        y2 = int(round(y1 + alto_region))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.ancho, x2)
        y2 = min(self.alto, y2)

        return x1, y1, x2, y2

    def _render(self):
        x1, y1, x2, y2 = self._region()
        region = self.imagen[y1:y2, x1:x2]

        if region.size == 0:
            region = self.imagen

        interpolacion = (
            cv2.INTER_AREA if self.escala < 1.0 else cv2.INTER_LINEAR
        )

        vista = cv2.resize(
            region,
            (self.ancho_vista, self.alto_vista),
            interpolation=interpolacion,
        )

        texto = f"{self.escala * 100:.0f}%  ({self.ancho}x{self.alto})"

        cv2.putText(
            vista,
            texto,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            vista,
            texto,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return vista

    def _zoom(self, acercar, x, y):
        x1, y1, _, _ = self._region()

        ix = x1 + x / self.escala
        iy = y1 + y / self.escala

        factor = self.FACTOR_ZOOM if acercar else 1.0 / self.FACTOR_ZOOM
        nueva = min(
            self.escala_max,
            max(self.escala_min, self.escala * factor),
        )

        if abs(nueva - self.escala) < 1e-6:
            return

        self.escala = nueva
        self.cx = ix - (x - self.ancho_vista / 2.0) / self.escala
        self.cy = iy - (y - self.alto_vista / 2.0) / self.escala

    def _evento(self, evento, x, y, flags, _parametro):
        if evento == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            self._zoom(delta > 0, x, y)

        elif evento == cv2.EVENT_LBUTTONDOWN:
            self._arrastrando = True
            self._ultimo = (x, y)

        elif evento == cv2.EVENT_MOUSEMOVE and self._arrastrando:
            dx = x - self._ultimo[0]
            dy = y - self._ultimo[1]
            self.cx -= dx / self.escala
            self.cy -= dy / self.escala
            self._ultimo = (x, y)

        elif evento == cv2.EVENT_LBUTTONUP:
            self._arrastrando = False

    def mostrar(self):
        cv2.namedWindow(self.nombre, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.nombre, self._evento)
        cv2.setWindowTitle(
            self.nombre,
            f"{self.nombre} | rueda: zoom | arrastrar: mover | "
            "1: 100% | r: ajustar | q: salir",
        )

        while True:
            cv2.imshow(self.nombre, self._render())
            tecla = cv2.waitKey(15) & 0xFF

            if tecla in (ord("q"), 27):
                break

            if tecla in (ord("r"), ord("0")):
                self.escala = self.escala_min
                self.cx = self.ancho / 2.0
                self.cy = self.alto / 2.0

            elif tecla == ord("1"):
                self.escala = min(
                    self.escala_max,
                    max(self.escala_min, 1.0),
                )
                self.cx = self.ancho / 2.0
                self.cy = self.alto / 2.0

            elif tecla in (ord("+"), ord("=")):
                self._zoom(True, self.ancho_vista // 2, self.alto_vista // 2)

            elif tecla in (ord("-"), ord("_")):
                self._zoom(False, self.ancho_vista // 2, self.alto_vista // 2)

        cv2.destroyWindow(self.nombre)


def mostrar(nombre, imagen):
    try:
        VisorImagen(nombre, imagen).mostrar()
    except cv2.error as error:
        print(f"No se pudo abrir la ventana ({error}).")
        print("La imagen procesada ya fue guardada en disco.")
