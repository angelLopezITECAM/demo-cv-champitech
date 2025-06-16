import shutil
import random
from pathlib import Path
import yaml

from loguru import logger
from tqdm import tqdm
import typer

# --- Asegúrate de que esta configuración es correcta ---
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    val_split_ratio: float = typer.Option(0.20, "--split-ratio", help="Proporción para validación y test del dataset local."),
    seed: int = typer.Option(42, "--seed", help="Semilla para la división aleatoria.")
):
    """
    Combina un dataset local (estructura simple) y un dataset externo 
    (estructura images/train, etc.) en un único dataset final.
    """
    logger.info("🚀 Iniciando la creación del dataset combinado (versión final)...")
    random.seed(seed)

    output_dir = PROCESSED_DATA_DIR / "final_dataset"

    # Limpieza y Creación de Directorios
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    subdirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # --- 1. PROCESAR DATASET LOCAL (PRIMORDIA) ---
    logger.info("Procesando dataset local 'primordia'...")
    local_source_dir = INTERIM_DATA_DIR / "primordia_date_split"
    local_img_dir = local_source_dir / "images"
    local_lbl_dir = local_source_dir / "labels"
    
    # Guardamos la ruta a la carpeta de imágenes explícitamente
    local_files = [{"label_path": path, "image_dir": local_img_dir} 
                   for path in local_lbl_dir.glob("*.txt") if path.read_text(encoding="utf-8").strip()]
    
    random.shuffle(local_files)
    train_ratio = 1 - val_split_ratio
    train_count = int(len(local_files) * train_ratio)
    val_count = int((len(local_files) - train_count) / 2)

    local_splits = {
        "train": local_files[:train_count],
        "val": local_files[train_count : train_count + val_count],
        "test": local_files[train_count + val_count:],
    }
    logger.info(f"Dataset 'primordia' dividido en {len(local_splits['train'])} train, {len(local_splits['val'])} val, y {len(local_splits['test'])} test.")

    # --- 2. PROCESAR DATASET EXTERNO (CON ESTRUCTURA DIFERENTE) ---
    logger.info("Procesando dataset externo 'm18k'...")
    external_source_dir = EXTERNAL_DATA_DIR / "m18ka"
    external_splits = {}
    for split in ["train", "val", "test"]:
        # Adaptamos la ruta para que coincida con la estructura del dataset externo
        img_dir_path = external_source_dir / "images" / split
        lbl_dir_path = external_source_dir / "labels" / split
        
        if not lbl_dir_path.exists():
            logger.warning(f"No se encontró el directorio de etiquetas para el split '{split}' en el dataset externo. Omitiendo.")
            external_splits[split] = []
            continue
            
        # Guardamos la ruta explícita a la carpeta de imágenes correcta
        external_splits[split] = [
            {"label_path": path, "image_dir": img_dir_path} 
            for path in lbl_dir_path.glob("*.txt")
        ]
    logger.info(f"Dataset externo con {len(external_splits['train'])} train, {len(external_splits['val'])} val, y {len(external_splits['test'])} test pre-definidos.")

    # --- 3. COPIAR Y UNIFICAR ---
    def copy_files(split_name, prefix, files_to_copy):
        for item in tqdm(files_to_copy, desc=f"Copiando {split_name} de {prefix}", leave=False, colour="green"):
            label_path = item["label_path"]
            image_dir = item["image_dir"] # Usamos la ruta explícita que guardamos antes
            base_name = label_path.stem
            
            possible_images = list(image_dir.glob(f"{base_name}.*"))
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            valid_images = [p for p in possible_images if p.suffix.lower() in image_extensions]
            
            if not valid_images:
                logger.warning(f"No se encontró imagen para la etiqueta {label_path}, se omitirá.")
                continue
            
            image_path = valid_images[0]
            new_filename_base = f"{prefix}_{base_name}"
            
            dest_label_path = output_dir / "labels" / split_name / f"{new_filename_base}.txt"
            dest_image_path = output_dir / "images" / split_name / f"{new_filename_base}{image_path.suffix}"

            shutil.copy(label_path, dest_label_path)
            shutil.copy(image_path, dest_image_path)

    for split in ["train", "val", "test"]:
        logger.info(f"Copiando y unificando el conjunto de '{split}'...")
        copy_files(split, "primordia", local_splits[split])
        copy_files(split, "public", external_splits[split])

    # --- 4. CREAR EL FICHERO data.yaml FINAL ---
    logger.info("Generando fichero 'data.yaml' final...")
    class_names = ['primordio']
    yaml_content = {
        'path': str(output_dir.resolve()), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'nc': len(class_names), 'names': class_names
    }
    with open(output_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, indent=2)

    logger.success(f"✅ ¡Dataset final combinado creado con éxito en '{output_dir}'!")

if __name__ == "__main__":
    app()