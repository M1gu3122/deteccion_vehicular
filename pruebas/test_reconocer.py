from reconocer_auto import predecir_marca_modelo
import cv2

pruebas = [
    r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos_v2\val\kia_sorento\kia_sorento_00030.jpg",
    r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos_v2\val\kia_sportage\kia_sportage_0005.jpg",
    r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos_v2\val\kia_sephia\kia_sephia_00019.jpg",
    r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos_v2\val\kia_rio\kia_rio_0009.jpg",
]

for ruta in pruebas:
    img = cv2.imread(ruta)
    if img is None:
        nombre = ruta.split("\\")[-1]
        print(f'{nombre:35s} -> ERROR: no carga')
        continue
    resultado = predecir_marca_modelo(img, conf_min=0.3)
    if resultado:
        etiqueta, conf = resultado
        print(f'{ruta.split("\\")[-1]:35s} -> {etiqueta:15s} ({conf:.3f})')
    else:
        nombre = ruta.split("\\")[-1]
        print(f'{nombre:35s} -> SIN CLASIFICAR')
" 2>&1