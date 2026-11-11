import shutil
from pathlib import Path
import os

BASE_DIR = Path(r"C:\Users\mglmo\OneDrive - Unibarranquilla\Desktop\Deteccion Vehicular")
DATASET_DIR = BASE_DIR / "dataset_autos"

def cuenta_imagenes(carpeta):
    if not os.path.exists(carpeta):
        return 0
    return len([f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

def reorganizar_dataset_completo():
    """
    Reorganiza el dataset completo para consistencia.
    Crea estructura correcta: carpetas con prefijo 'kia_' a nivel raíz.
    """
    print("=" * 80)
    print("REORGANIZACIÓN COMPLETA DEL DATASET")
    print("=" * 80)

    # Crear backup
    backup_dir = DATASET_DIR / "backup_completo_A_KJLZ1"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(DATASET_DIR, backup_dir)
    print(f"\n✓ Backup creado en: {backup_dir}")

    # Obtener nombres correctos de clases
    nombres_clases_correctos = [
        'amanti', 'forte', 'optima', 'picanto',
        'rio', 'sedona', 'sephia', 'sorento',
        'soul', 'spectra', 'sportage'
    ]

    print("\n1. Estructura original (se detecta):")
    print("-" * 80)

    # Imágenes por carpeta
    for clase in nombres_clases_correctos:
        ruta_raiz = DATASET_DIR / clase
        ruta_train = DATASET_DIR / "train" / f"kia_{clase}"
        ruta_val = DATASET_DIR / "val" / f"kia_{clase}"

        imgs_raiz = cuenta_imagenes(ruta_raiz)
        imgs_train = cuenta_imagenes(ruta_train)
        imgs_val = cuenta_imagenes(ruta_val)
        total = imgs_raiz + imgs_train + imgs_val

        print(f"  {clase:12} - R = {imgs_raiz:3d} - T = {imgs_train:3d} - V = {imgs_val:3d} = {total:4d}")

    print("\n2. NUEVA ESTRUCTURA RECOMENDADA:")
    print("-" * 80)
    print("  Todas las carpetas de clases AL MISMO NIVEL con prefijo 'kia_':")
    print("    - dataset_autos/kia_amanti/")
    print("    - dataset_autos/kia_forte/")
    print("    ...")

    print("\n3. Procediendo con reorganización:")
    print("-")

    # NUEVA ESTRUCTURA: Mover todas las carpetas con 'kia_' al nivel raíz
    print("\n  Paso 1: Reorganizando carpetas...")
    respaldo_train = DATASET_DIR / "train_backup_01"
    respaldo_val = DATASET_DIR / "val_backup_01"

    # Mover train a backup temporal
    if respaldo_train.exists():
        shutil.rmtree(respaldo_train)
    shutil.move(str(DATASET_DIR / "train"), str(respaldo_train))

    # Mover val a backup temporal
    if respaldo_val.exists():
        shutil.rmtree(respaldo_val)
    shutil.move(str(DATASET_DIR / "val"), str(respaldo_val))

    print(f"  ✓ Train movido a: {respaldo_train}")
    print(f"  ✓ Val movido a: {respaldo_val}")

    # Crear carpetas con nombres consistentes
    print("\n  Paso 2: Creando carpetas con prefijo 'kia_'...")
    for clase in nombres_clases_correctos:
        nueva_carpeta = DATASET_DIR / f"kia_{clase}"
        nueva_carpeta.mkdir(exist_ok=True)
        print(f"  ✓ {clase:12} -> {nueva_carpeta.name}")

    # Mover contenido a nuevas carpetas
    print("\n  Paso 3: Copiando imágenes a nuevas carpetas...")
    for clase in nombres_clases_correctos:
        ruta_raiz = DATASET_DIR / clase
        ruta_train_orig = respaldo_train / f"kia_{clase}"
        ruta_val_orig = respaldo_val / f"kia_{clase}"

        # Agregar imágenes de raíz
        if ruta_raiz.exists():
            archivos = [f for f in os.listdir(ruta_raiz) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if archivos:
                shutil.copytree(ruta_raiz, DATASET_DIR / f"kia_{clase}", dirs_exist_ok=True)
                print(f"  ✓ Ruta raíz: {clase:12} ({len(archivos)} imágenes)")

        # Agregar imágenes de train
        if ruta_train_orig.exists():
            archivos = [f for f in os.listdir(ruta_train_orig) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if archivos:
                shutil.copytree(ruta_train_orig, DATASET_DIR / f"kia_{clase}", dirs_exist_ok=True)
                print(f"  ✓ Train:     {clase:12} ({len(archivos)} imágenes)")

        # Agregar imágenes de val
        if ruta_val_orig.exists():
            archivos = [f for f in os.listdir(ruta_val_orig) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if archivos:
                shutil.copytree(ruta_val_orig, DATASET_DIR / f"kia_{clase}", dirs_exist_ok=True)
                print(f"  ✓ Val:       {clase:12} ({len(archivos)} imágenes)")

    # Crear directorios train y val vacíos
    print("\n  Paso 4: Creando carpetas train y val...")
    DATASET_DIR / "train" / "train_images"  # Purga de YOLO
    DATASET_DIR / "val" / "val_images"     # Purga de YOLO
    print("  ✓ Carpetas train/ y val/ preparadas")

    # Informe final
    print("\n" + "=" * 80)
    print("ESTRUCTURA FINAL")
    print("=" * 80)
    print("\nNuevo directorio: dataset_autos/")
    print("  ├── kia_amanti/")
    print("  ├── kia_forte/")
    print("  ├── kia_optima/")
    print("  ├── ...")
    print("  ├── kia_sorento/")
    print("  ├── train/")
    print("  └── val/")

    print("\nConteo final por clase:")
    total_imagenes = 0
    for clase in nombres_clases_correctos:
        ruta = DATASET_DIR / f"kia_{clase}"
        imgs = cuenta_imagenes(ruta)
        print(f"  {clase:12}: {imgs:4d} imágenes")
        total_imagenes += imgs

    print("=" * 80)
    print("DATASET REORGANIZADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nTotal de imágenes: {total_imagenes}")
    print(f"Backup original: {backup_dir}")
    print("\nSiguiente paso: Crear data.yaml (ya se hizo)")
    print("   Pasos: python entrenar_clasificador.py")
    print("")

if __name__ == "__main__":
    try:
        reorganizar_dataset_completo()
    except Exception as e:
        print(f"\nError en reorganización: {e}")
        import traceback
        traceback.print_exc()