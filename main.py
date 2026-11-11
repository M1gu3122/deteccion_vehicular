from config import IMAGEN_ENTRADA, IMAGEN_SALIDA
from reconocedor import ReconocedorPlacas


def main():
    reconocedor = ReconocedorPlacas()

    resultados = reconocedor.procesar_imagen(
        IMAGEN_ENTRADA
    )

    if not resultados:
        print("No se detectaron placas.")
    else:
        for numero, resultado in enumerate(resultados, start=1):
            print(f"\nPlaca detectada #{numero}")
            print(f"Texto original: {resultado['texto_original']}")
            print(f"Placa normalizada: {resultado['placa']}")
            print(
                f"Confianza YOLO: "
                f"{resultado['confianza_yolo']:.2f}"
            )
            print(
                f"Confianzas OCR: "
                f"{resultado['confianzas_ocr']}"
            )

            if not resultado["placa"]:
                print("No se pudo reconocer el texto.")

    print(f"\nImagen procesada: {IMAGEN_SALIDA}")


if __name__ == "__main__":
    main()