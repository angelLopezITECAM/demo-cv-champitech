import typer
from ultralytics import YOLO
from pathlib import Path
from loguru import logger

app = typer.Typer()

@app.command()
def predict(
    weights_path: Path = typer.Option(
        ..., 
        "--weights-path", 
        help="Ruta al fichero de pesos del modelo entrenado (ej: models/experimento/weights/best.pt)."
    ),
    input_path: Path = typer.Option(
        ..., # <-- CAMBIO CLAVE: Usar typer.Option en lugar de typer.Typer
        "--input-path", 
        help="Ruta a la imagen, vídeo o carpeta de imágenes para la predicción."
    ),
    output_dir: Path = typer.Option(
        "reports/figures/", 
        "--output-dir", 
        help="Directorio donde se guardarán las imágenes con las predicciones."
    )
):
    """
    Usa un modelo YOLO entrenado para hacer una predicción sobre una nueva imagen,
    vídeo o carpeta.
    """
    logger.info(f"Cargando modelo desde: {weights_path}")
    if not weights_path.exists():
        logger.error("El fichero de pesos especificado no existe.")
        raise typer.Exit(code=1)
        
    model = YOLO(weights_path)

    logger.info(f"Realizando predicción sobre: {input_path}")
    if not input_path.exists():
        logger.error("La ruta de entrada especificada no existe.")
        raise typer.Exit(code=1)

    # Realizar la predicción
    results = model.predict(source=str(input_path))
    
    # El resultado es una lista
    if not results:
        logger.warning("El modelo no devolvió resultados.")
        raise typer.Exit()
        
    result = results[0]
    
    # Crear el directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predicted_{input_path.name}"

    logger.info(f"Guardando imagen con predicciones en: {output_path}")
    result.save(filename=str(output_path))
    
    logger.info("Cajas de detección encontradas:")
    if result.boxes:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            coords = [round(c) for c in box.xyxy[0].tolist()]
            
            logger.info(f"- Clase: {class_name} (ID: {class_id}) | Confianza: {confidence:.2f} | Coordenadas: {coords}")
    else:
        logger.info("No se encontraron cajas de detección en la imagen.")

    logger.success("✅ Predicción completada.")


if __name__ == "__main__":
    app()