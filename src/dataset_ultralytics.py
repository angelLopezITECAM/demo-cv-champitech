# Contenido de src/data/dataset.py para la Opción 1 (la recomendada)
import shutil
import random
from pathlib import Path
import yaml

from loguru import logger
from tqdm import tqdm
import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    data_subdir: str = typer.Option("primordia", help="Subdirectorio específico del dataset."),
    val_split_ratio: float = typer.Option(0.20, "--split-ratio", help="Proporción de validación."),
    image_ext: str = typer.Option(".webp", "--image-ext", help="Extensión de las imágenes."),
    seed: int = typer.Option(42, "--seed", help="Semilla para la división aleatoria.")
):
    logger.info("🚀 Iniciando la creación del dataset para YOLO (método de copia)...")
    random.seed(seed)

    raw_dir = RAW_DATA_DIR / data_subdir
    processed_dir = PROCESSED_DATA_DIR / data_subdir
    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"

    if processed_dir.exists():
        logger.warning(f"Limpiando directorio antiguo en {processed_dir}...")
        shutil.rmtree(processed_dir)

    logger.info("Creando nueva estructura de directorios en 'processed'...")
    train_images_path = processed_dir / "images" / "train"
    val_images_path = processed_dir / "images" / "val"
    train_labels_path = processed_dir / "labels" / "train"
    val_labels_path = processed_dir / "labels" / "val"
    train_images_path.mkdir(parents=True, exist_ok=True)
    val_images_path.mkdir(parents=True, exist_ok=True)
    train_labels_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)

    valid_label_files = [f for f in raw_labels_dir.glob("*.txt") if f.read_text(encoding="utf-8").strip()]
    logger.info(f"Encontrados {len(valid_label_files)} ficheros de etiquetas válidos.")
    random.shuffle(valid_label_files)
    
    split_point = int(len(valid_label_files) * (1 - val_split_ratio))
    train_files = valid_label_files[:split_point]
    val_files = valid_label_files[split_point:]

    logger.success(f"División de datos: {len(train_files)} entrenamiento | {len(val_files)} validación.")

    def copy_files(file_list, split_name):
        logger.info(f"Copiando ficheros de {split_name}...")
        for label_path in tqdm(file_list, desc=f"Copiando {split_name}"):
            base_name = label_path.stem
            image_path = raw_images_dir / f"{base_name}{image_ext}"
            
            shutil.copy(label_path, processed_dir / "labels" / split_name)
            if image_path.exists():
                shutil.copy(image_path, processed_dir / "images" / split_name)
            else:
                logger.warning(f"No se encontró la imagen: {image_path}")

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    logger.info("Generando fichero 'data.yaml'...")
    class_names = ['primordio'] # Modifica si tienes más clases
    yaml_content = {
        'path': str(processed_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(processed_dir / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, indent=2)

    logger.success("✅ ¡Proceso completado! Dataset listo para el 'train.py' de Ultralytics.")

if __name__ == "__main__":
    app()