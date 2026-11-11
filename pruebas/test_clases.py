from ultralytics import YOLO

ruta_modelo = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\runs\auto_cls\weights\best.pt"

print(f"Cargando modelo desde: {ruta_modelo}")

try:
    modelo = YOLO(ruta_modelo)
    
    print("\n=== Clases definidas en el modelo ===")
    print("IDs y nombres:")
    for i, name in sorted(modelo.names.items(), key=lambda x: x[1].lower()):
        print(f"  {i}: {name}")
    
    # Verificar si hay data.txt y sus categorías
    import os
    data_path = r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\data.yaml" if os.path.exists(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular\dataset_autos\data.yaml") else None
    
    if data_path:
        print(f"\n=== data.yaml encontrado ===")
        with open(data_path, 'r') as f:
            content = f.read()
            # Buscar las líneas de clases
            if 'names:' in content:
                import re
                classes_match = re.search(r'names:\s*\[[^\]]*\]', content)
                if classes_match:
                    print(f"Clases del data.yaml: {classes_match.group(0)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()