import shutil
import random
from pathlib import Path
import pandas as pd  # Necesitaremos pandas para manejar los CSVs

from loguru import logger
import typer

# Asumimos que estas variables apuntan a las carpetas data/raw y data/processed
# según la configuración de tu proyecto Cookiecutter.
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    
    data_subdir: str = typer.Option("primordia", help="Subdirectorio específico del dataset."),
    val_split_ratio: float = typer.Option(0.20, "--split-ratio", help="Proporción de datos para el conjunto de validación."),
    image_ext: str = typer.Option(".webp", "--image-ext", help="Extensión de los ficheros de imagen."),
    seed: int = typer.Option(42, "--seed", help="Semilla para la división aleatoria de los datos.")
):
    """
    Crea ficheros CSV (train.csv y val.csv) en la carpeta PROCESSED
    que mapean los ficheros de datos en la carpeta RAW, sin duplicar datos.
    """
    # --- 1. CONFIGURACIÓN Y RUTAS ---
    logger.info("🚀 Creando los manifiestos CSV del dataset...")
    random.seed(seed)

    raw_dir = RAW_DATA_DIR / data_subdir
    processed_dir = PROCESSED_DATA_DIR / data_subdir
    
    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"
    raw_metadata_dir = raw_dir / "data" # Tus ficheros .json

    logger.info(f"Directorio de datos RAW: {raw_dir}")
    logger.info(f"Directorio de datos PROCESSED: {processed_dir}")

    # Asegurarse de que el directorio de salida exista
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. DESCUBRIR TODOS LOS DATOS ---
    logger.info(f"Buscando etiquetas en {raw_labels_dir}...")
    all_label_files = list(raw_labels_dir.glob("*.txt"))

    dataset_records = []
    for label_path in all_label_files:
        base_name = label_path.stem
        image_path = raw_images_dir / f"{base_name}{image_ext}"
        metadata_path = raw_metadata_dir / f"{base_name}.json"

        # Solo añadir el registro si la imagen correspondiente existe
        if image_path.exists():
            dataset_records.append({
                "image_path": str(image_path.relative_to(RAW_DATA_DIR.parent)),
                "label_path": str(label_path.relative_to(RAW_DATA_DIR.parent)),
                "metadata_path": str(metadata_path.relative_to(RAW_DATA_DIR.parent))
            })
        else:
            logger.warning(f"No se encontró la imagen para la etiqueta {label_path.name}. Se omitirá.")

    if not dataset_records:
        logger.error("No se encontraron pares de imagen/etiqueta válidos. Abortando.")
        raise typer.Exit()
        
    logger.info(f"Se encontraron {len(dataset_records)} registros completos (imagen, etiqueta, metadatos).")

    # --- 3. MEZCLAR Y DIVIDIR LOS REGISTROS ---
    random.shuffle(dataset_records)
    
    split_point = int(len(dataset_records) * (1 - val_split_ratio))
    train_records = dataset_records[:split_point]
    val_records = dataset_records[split_point:]

    logger.success(f"División de datos: {len(train_records)} entrenamiento | {len(val_records)} validación.")

    # --- 4. CREAR Y GUARDAR LOS DATAFRAMES ---
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    
    train_csv_path = processed_dir / "train.csv"
    val_csv_path = processed_dir / "val.csv"

    logger.info(f"Guardando fichero de entrenamiento en: {train_csv_path}")
    train_df.to_csv(train_csv_path, index=False)
    
    logger.info(f"Guardando fichero de validación en: {val_csv_path}")
    val_df.to_csv(val_csv_path, index=False)
    
    logger.success("✅ ¡Proceso completado! Manifiestos CSV creados en 'data/processed/primordia'")


if __name__ == "__main__":
    # Para ejecutar desde la terminal, necesitarás instalar pandas:
    # pip install pandas
    app()