from ultralytics import YOLO
import glob
import os

ruta_modelo = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\runs\auto_cls\weights\best.pt"

print(f"Cargando modelo desde: {ruta_modelo}")

try:
    modelo = YOLO(ruta_modelo)
    
    # Buscar imágenes de diferentes clases
    car_class = 'picanto'
    ruta_clase1 = fr"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\{car_class}"
    
    # Encontrar primer archivo de kia_sorento
    clase2 = 'kia_sorento'
    ruta_clase2 = fr"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\{clase2}"
    files = glob.glob(os.path.join(ruta_clase2, "*.jpg"))
    
    if files:
        ruta_img1 = files[0]
        ruta_img2 = os.path.join(ruta_clase1, glob.glob(os.path.join(ruta_clase1, "*.jpg"))[0])
        
        print(f"\n=== Probando imagen de {clase2} ===")
        resultado1 = modelo.predict(ruta_img1, verbose=False)[0]
        
        if resultado1.probs is not None and hasattr(resultado1.probs, 'top5'):
            print("Probabilidades (top 5):")
            for i, conf in zip(resultado1.probs.top5, resultado1.probs.top5conf):
                print(f"  {modelo.names[int(i)]}: {float(conf):.6f}")
        elif resultado1.probs is not None:
            print(f"Resultado type: {type(resultado1.probs.top1)}")
            print(f"Confianza picanto: {float(resultado1.probs.top1conf) if hasattr(resultado1.probs, 'top1conf') else 'N/A'}")
        
        print(f"\n=== Probando imagen de {car_class} ===")
        resultado2 = modelo.predict(ruta_img2, verbose=False)[0]
        
        if resultado2.probs is not None and hasattr(resultado2.probs, 'top5'):
            print("Probabilidades (top 5):")
            for i, conf in zip(resultado2.probs.top5, resultado2.probs.top5conf):
                print(f"  {modelo.names[int(i)]}: {float(conf):.6f}")
        elif resultado2.probs is not None:
            print(f"Resultado type: {type(resultado2.probs.top2)}")
            print(f"Confianza {car_class}: {float(resultado2.probs.top1conf) if hasattr(resultado2.probs, 'top1conf') else 'N/A'}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()