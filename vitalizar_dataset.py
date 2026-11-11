import shutil
from pathlib import Path
import os

BASE_DIR = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
DATASET_DIR = BASE_DIR / "dataset_autos"

def crear_data_yaml():
    """Crear un archivo data.yaml configurado correctamente"""
    print("Creando data.yaml...")

    # Obtener nombres de clases
    carpetas_train = [d for d in os.listdir(DATASET_DIR / "train") if os.path.isdir(DATASET_DIR / "train" / d) and "kia_" in d]
    carpetas_nombres = sorted([c.replace("kia_", "") for c in carpetas_train])

    print(f"Clases detectadas: {carpetas_nombres}")

    yaml_content = f"""
# Dataset de Clasificación Kia
path: {DATASET_DIR.absolute()}
train: train
val: val
names:
"""

    for nombre in carpetas_nombres:
        yaml_content += f"  {carpetas_nombres.index(nombre)}: {nombre}\n"

    archivo_yaml = BASE_DIR / "dataset_autos" / "data.yaml"
    with open(archivo_yaml, "w") as f:
        f.write(yaml_content)

    print(f"data.yaml creado en: {archivo_yaml}")

def reorganizar_dataset():
    """Reorganizar carpetas para consistencia"""
    print("\nReorganizando dataset...")

    # Crear backup del dataset actual
    backup_dir = DATASET_DIR / "backup_desorganizado"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(DATASET_DIR, backup_dir)

    # Ráiz de carpetas con prefijo "kia_"
    carpetas = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(DATASET_DIR / d) and "kia_" in d]

    # Para cada clase, comprobar direcciones
    for carpeta in carpetas:
        nombre_real = carpeta.replace("kia_", "")

        # Lista de carpetas existentes en las direcciones raíz y train/
        carpetas_raiz = [d.replace("kia_", "") for d in os.listdir(DATASET_DIR) if os.path.isdir(DATASET_DIR / d)]
        carpetas_train_class = [d for d in os.listdir(DATASET_DIR / "train") if d.startswith("kia_")]

        # Buscar correspondencia por número de archivos
        if nombre_real in carpetas_raiz:
            print(f"[OK] {nombre_real}: Ya existe en raiz")

        if any(d.startswith("kia_" + nombre_real) for d in carpetas_train_class):
            print(f"[OK] {nombre_real}: Ya existe en train/")

    print("\nDataset reorganizado correctamente")

if __name__ == "__main__":
    crear_data_yaml()
    reorganizar_dataset()
    print("\n=== REQUERIMIENTOS PARA REENTRENAMIENTO ===")
    print("1. Verificar que data.yaml está completo y curado")
    print("2. Confirmar consistencia de nombres de carpetas")
    print("3. Ejecutar: python entrenar_clasificador.py")