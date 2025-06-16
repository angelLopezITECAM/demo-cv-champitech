import typer
from pathlib import Path
import json
from PIL import Image, ImageDraw
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import shutil

app = typer.Typer()

def date_to_color(day: int, max_days: int = 30) -> tuple:
    """Convierte un número de día en un color degradado de azul a rojo."""
    if day < 0: day = 0
    if day > max_days: day = max_days
    normalized_day = day / max_days
    red = int(255 * normalized_day)
    blue = int(255 * (1 - normalized_day))
    return (red, 0, blue)

@app.command()
def add_date_visual_indicator(
    input_dir: Path = typer.Option("data/raw/primordia", help="Directorio con 'images', 'labels' y 'data' (JSONs)."),
    output_dir: Path = typer.Option("data/interim/primordia_with_date", help="Directorio de salida para las nuevas imágenes y etiquetas."),
    max_days: int = typer.Option(20, help="Número máximo de días para la escala de color (afecta al degradado).")
):
    """
    Lee un dataset, calcula el día de cultivo a partir de los metadatos JSON
    y añade un indicador visual de la fecha a cada imagen.
    """
    logger.info(f"Iniciando el proceso para añadir indicador de fecha a las imágenes...")
    
    images_in_dir = input_dir / "images"
    json_in_dir = input_dir / "data"
    labels_in_dir = input_dir / "labels"

    images_out_dir = output_dir / "images"
    labels_out_dir = output_dir / "labels"
    
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(images_in_dir.glob("*.*"))
    
    for img_path in tqdm(image_paths, desc="Procesando imágenes"):
        base_name = img_path.stem
        json_path = json_in_dir / f"{base_name}.json"
        label_path = labels_in_dir / f"{base_name}.txt"

        if not json_path.exists():
            logger.warning(f"No se encontró JSON para {img_path.name}. Se omitirá.")
            continue
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            start_date = datetime.strptime(metadata['dia_entrada'], "%Y-%m-%d")
            image_date = datetime.strptime(metadata['fecha'], "%Y-%m-%d")
            day_of_cultivation = (image_date - start_date).days
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Error procesando {json_path.name}: {e}. Se omitirá.")
            continue

        with Image.open(img_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            color = date_to_color(day_of_cultivation, max_days)
            bar_height = 20
            draw.rectangle([0, 0, img.width, bar_height], fill=color)
            img.save(images_out_dir / img_path.name)

        if label_path.exists():
            shutil.copy(label_path, labels_out_dir / label_path.name)

    logger.success(f"Proceso completado. Nuevas imágenes guardadas en {images_out_dir}")

if __name__ == "__main__":
    app()