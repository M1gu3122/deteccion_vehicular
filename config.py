from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent

# Directorios
MODELOS_DIR = BASE_DIR / "modelos"
IMAGENES_DIR = BASE_DIR / "imagenes"
RESULTADOS_DIR = BASE_DIR / "resultados"

# Archivos
MODELO_PLACAS = MODELOS_DIR / "best.pt"
IMAGEN_ENTRADA = IMAGENES_DIR / "auto2.jpg"
IMAGEN_SALIDA = RESULTADOS_DIR / "imagen_procesada.jpg"

# Parámetros de detección
CONFIANZA_MINIMA = 0.70

# Margen añadido alrededor de la placa
MARGEN_HORIZONTAL = 0.08
MARGEN_VERTICAL = 0.15

# Idioma del OCR
IDIOMA_OCR = "en"