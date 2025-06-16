import typer
from ultralytics import YOLO

app = typer.Typer()

@app.command()
def main(
    # Mantenemos solo estos dos parámetros configurables
    model_base_name: str = typer.Option(
        "yolov8n", 
        "--model", 
        help="Nombre base del modelo YOLO a usar (ej: yolov8n, yolov8s)."
    ),
    num_epochs: int = typer.Option(
        150, 
        "--epochs", 
        help="Número de épocas para el entrenamiento."
    )
):
    """ data_yaml_path = './data/processed/final_dataset/data.yaml' """
    data_yaml_path = './data/interim/primordia_split/data.yaml'
    img_size = 640
    experiment_name = f"{model_base_name}_{num_epochs}epochs"
    model = YOLO(f"{model_base_name}.pt")

    # --- 4. Entrenamiento del Modelo ---
    typer.echo(f"🚀 Iniciando entrenamiento del modelo '{model_base_name}' por {num_epochs} épocas...")
    typer.echo(f"Los resultados se guardarán en: models/{experiment_name}")

    model.train(
        data=data_yaml_path,  # Usa la variable fija
        epochs=num_epochs,
        imgsz=img_size,      # Usa la variable fija
        project='models',
        name=experiment_name,
        exist_ok=True
    )
    
    typer.secho("✅ Entrenamiento finalizado con éxito.", fg=typer.colors.GREEN)

# Punto de entrada para ejecutar la aplicación
if __name__ == "__main__":
    app()