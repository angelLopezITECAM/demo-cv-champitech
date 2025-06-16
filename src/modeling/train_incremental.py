# train_incremental.py
import typer
from pathlib import Path
import json
from datetime import datetime
import shutil
from collections import defaultdict
from ultralytics import YOLO
import yaml
from loguru import logger

# Asumiendo que las rutas se importan desde un src/config.py centralizado
# (Esta es la mejor práctica recomendada)

PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

app = typer.Typer()

def get_day_of_cultivation(json_path: Path) -> int:
    """Calcula el día de cultivo desde un fichero JSON de metadatos."""
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    start_date = datetime.strptime(metadata['dia_entrada'], "%Y-%m-%d")
    image_date = datetime.strptime(metadata['fecha'], "%Y-%m-%d")
    return (image_date - start_date).days

@app.command()
def main(
    model_base_name: str = typer.Option("yolov8m", "--model", help="Modelo YOLO base para el primer entrenamiento."),
    num_epochs: int = typer.Option(50, "--epochs", help="Número de épocas para cada paso de entrenamiento."),
    img_size: int = typer.Option(640, "--imgsz", help="Tamaño de imagen para el entrenamiento."),
    batch_size: int = typer.Option(8, "--batch-size", help="Tamaño del batch para el entrenamiento. ¡Redúcelo si te quedas sin memoria!"),
    use_amp: bool = typer.Option(True, "--amp/--no-amp", help="Usar Automatic Mixed Precision (AMP) para ahorrar memoria."),
    data_subdir: str = typer.Option("primordia", help="Subdirectorio en data/raw que contiene los datos."),
):
    """
    Realiza un entrenamiento incremental día a día.
    """
    logger.info("🚀 Iniciando orquestador de entrenamiento temporal incremental.")
    
    logger.info("Mapeando imágenes a sus días de cultivo...")
    raw_images_dir = RAW_DATA_DIR / data_subdir / "images"
    raw_labels_dir = RAW_DATA_DIR / data_subdir / "labels"
    raw_metadata_dir = RAW_DATA_DIR / data_subdir / "data"
    
    files_by_day = defaultdict(list)
    
    all_label_files = list(raw_labels_dir.rglob("*.txt"))

    if not all_label_files:
        logger.error(f"No se encontraron ficheros .txt en {raw_labels_dir.resolve()}. Revisa la ruta.")
        raise typer.Exit()

    for label_path in all_label_files:
        base_name = label_path.stem
        json_path = raw_metadata_dir / f"{base_name}.json"
        image_path = next(raw_images_dir.glob(f"{base_name}.*"), None)

        if json_path.exists() and image_path:
            try:
                day = get_day_of_cultivation(json_path)
                files_by_day[day].append({"image": image_path, "label": label_path})
            except Exception as e:
                logger.warning(f"Omitiendo {label_path.name} por error: {e}")
        else:
            logger.warning(f"Omitiendo {label_path.name} por falta de imagen o JSON.")

    sorted_days = sorted(files_by_day.keys())
    if len(sorted_days) < 2:
        logger.error("Se necesitan datos de al menos dos días diferentes para entrenar.")
        raise typer.Exit()
    logger.success(f"Datos encontrados y clasificados para los días: {sorted_days}")

    # --- 2. Bucle de Entrenamiento Principal ---
    last_model_weights = Path(f"{model_base_name}.pt")

    for i in range(len(sorted_days) - 1):
        train_days = sorted_days[:i + 1]
        validation_day = sorted_days[i + 1]
        experiment_name = f"train_d{'-'.join(map(str, train_days))}_val_d{validation_day}"
        logger.info(f"--- Iniciando Iteración: {experiment_name} ---")

        # --- 3. Preparación de Datos para la Iteración ---
        # ... (Esta sección no cambia)
        run_data_dir = INTERIM_DATA_DIR / "temporal_runs" / experiment_name
        if run_data_dir.exists():
            shutil.rmtree(run_data_dir)
        train_img_dir = run_data_dir / "images" / "train"
        val_img_dir = run_data_dir / "images" / "val"
        train_lbl_dir = run_data_dir / "labels" / "train"
        val_lbl_dir = run_data_dir / "labels" / "val"
        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)
        for day in train_days:
            for file_pair in files_by_day[day]:
                shutil.copy(file_pair["image"], train_img_dir)
                shutil.copy(file_pair["label"], train_lbl_dir)
        for file_pair in files_by_day[validation_day]:
            shutil.copy(file_pair["image"], val_img_dir)
            shutil.copy(file_pair["label"], val_lbl_dir)
        yaml_path = run_data_dir / "data.yaml"
        yaml_content = {
            'path': str(run_data_dir.resolve()), 'train': 'images/train', 'val': 'images/val',
            'nc': 1, 'names': ['primordio']
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        # --- 4. Entrenamiento Incremental ---
        model = YOLO(last_model_weights)
        
        logger.info(f"Entrenando con pesos de: {last_model_weights}, batch_size={batch_size}, imgsz={img_size}, amp={use_amp}")
        # --- LLAMADA A TRAIN ACTUALIZADA ---
        model.train(
            data=str(yaml_path.resolve()),
            epochs=num_epochs,
            imgsz=img_size,
            batch=batch_size, # Pasamos el batch size
            amp=use_amp,      # Activamos o desactivamos AMP
            project=str(MODELS_DIR.resolve()),
            name=experiment_name,
            exist_ok=True
        )

        last_model_weights = MODELS_DIR / experiment_name / "weights" / "best.pt"
        if not last_model_weights.exists():
            logger.error(f"No se encontraron los pesos 'best.pt'. Abortando.")
            raise typer.Exit()
        logger.success(f"✅ Iteración completada. Modelo guardado en {last_model_weights}")

    logger.info("🎉 ¡Proceso de entrenamiento incremental finalizado con éxito! 🎉")

if __name__ == "__main__":
    app()