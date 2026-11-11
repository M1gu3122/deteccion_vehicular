from ultralytics import YOLO
import sys

ruta_modelo = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\runs\auto_cls\weights\best.pt"

print(f"Cargando modelo desde: {ruta_modelo}")

try:
    modelo = YOLO(ruta_modelo)
    
    # Probar con una imagen de kia_sorento
    ruta_entrada = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\kia_sorento\sorento_calle_00030.jpg"
    ruta_sorento = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\kia_sorento\sorento_calle_00030.jpg"
    
    print(f"\nProbando con imagen de kia_sorento:")
    resultado = modelo.predict(ruta_sorento, verbose=True, conf=0.1)[0]
    
    if resultado.probs is not None:
        print(f"\nResultado type: {type(resultado.probs.top1)}")
        if isinstance(resultado.probs.top1, (list, tuple)):
            top1 = resultado.probs.top1[0]
            top1conf = resultado.probs.top1conf[0]
        else:
            # Ya tiene el índice
            top1 = resultado.probs.top1
            top1conf = resultado.probs.top1conf
        
        print(f"\nProbabilidades:")
        if isinstance(resultado.probs, (list, tuple)):
            for i, conf in zip(resultado.probs[0], resultado.probs[1]):
                print(f"  {modelo.names[int(i)]}: {float(conf):.4f}")
        else:
            print(f"  top1: {top1}, top1conf: {top1conf}")
        
        top1_name = modelo.names[int(top1)]
        top1_conf = float(top1conf)
        
        print(f"\nPredicción principal: {top1_name} ({top1_conf:.4f})")
    
    # Probar con una imagen de picanto
    print(f"\nProbando con imagen de picanto:")
    resultado2 = modelo.predict(ruta_entrada, verbose=True, conf=0.1)[0]
    
    if resultado2.probs is not None:
        print(f"\nProbabilidades:")
        for i, conf in zip(resultado2.probs.top1, resultado2.probs.top1conf):
            print(f"  {modelo.names[int(i)]}: {float(conf):.4f}")
        
        top1_name = modelo.names[int(resultado2.probs.top1)]
        top1_conf = float(resultado2.probs.top1conf)
        
        print(f"\nPredicción principal: {top1_name} ({top1_conf:.4f})")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()