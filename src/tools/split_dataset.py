import typer
from pathlib import Path
import random
import shutil
from loguru import logger
from tqdm import tqdm

app = typer.Typer()

@app.command()
def split_data(
    input_dir: Path = typer.Option(..., "--input-dir", help="Directorio con los datos originales (con carpetas 'images' y 'labels')."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directorio donde se guardará el dataset dividido."),
    ratios: str = typer.Option("0.7,0.2,0.1", "--ratios", help="Proporciones para train,valid,test separadas por comas."),
    seed: int = typer.Option(42, "--seed", help="Semilla para la división aleatoria.")
):
    """
    Divide un dataset de imágenes y etiquetas en conjuntos de train, valid y test.
    """
    logger.info(f"Iniciando la división del dataset en '{input_dir}'...")
    random.seed(seed)
    
    # --- 1. Validar y Parsear Ratios ---
    try:
        train_r, val_r, test_r = [float(x) for x in ratios.split(',')]
        assert abs(sum([train_r, val_r, test_r]) - 1.0) < 1e-8, "Los ratios deben sumar 1."
    except (ValueError, AssertionError) as e:
        logger.error(f"Error en los ratios: {e}")
        raise typer.Exit()

    # --- 2. Descubrir Ficheros ---
    labels_dir = input_dir / "labels"
    images_dir = input_dir / "images"
    
    all_label_paths = [p for p in labels_dir.glob("*.txt") if p.read_text(encoding="utf-8").strip()]
    if not all_label_paths:
        logger.error(f"No se encontraron ficheros de etiquetas en {labels_dir}")
        raise typer.Exit()

    logger.info(f"Encontrados {len(all_label_paths)} pares de imagen/etiqueta.")
    random.shuffle(all_label_paths)

    # --- 3. Calcular y Asignar Divisiones ---
    total_count = len(all_label_paths)
    train_count = int(total_count * train_r)
    val_count = int(total_count * val_r)
    
    splits = {
        "train": all_label_paths[:train_count],
        "val": all_label_paths[train_count : train_count + val_count],
        "test": all_label_paths[train_count + val_count :],
    }

    logger.success(f"División calculada: {len(splits['train'])} Train | {len(splits['val'])} Val | {len(splits['test'])} Test")

    # --- 4. Limpiar y Crear Carpetas de Destino ---
    if output_dir.exists():
        logger.warning(f"El directorio de salida {output_dir} ya existe. Se eliminará.")
        shutil.rmtree(output_dir)
        
    for split_name in splits.keys():
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # --- 5. Copiar Ficheros ---
    for split_name, label_paths in splits.items():
        logger.info(f"Copiando ficheros de '{split_name}'...")
        for label_path in tqdm(label_paths, desc=f"Copiando {split_name}"):
            # Encontrar imagen correspondiente con cualquier extensión
            base_name = label_path.stem
            possible_images = list(images_dir.glob(f"{base_name}.*"))
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            valid_images = [p for p in possible_images if p.suffix.lower() in image_extensions]

            if valid_images:
                image_path = valid_images[0]
                # Copiar ambos ficheros
                shutil.copy(label_path, output_dir / "labels" / split_name)
                shutil.copy(image_path, output_dir / "images" / split_name)
            else:
                logger.warning(f"No se encontró imagen para la etiqueta {label_path.name}, se omitirá.")

    logger.success(f"¡Dataset dividido con éxito en '{output_dir}'!")


if __name__ == "__main__":
    app()