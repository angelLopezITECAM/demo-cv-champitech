# Contenido para run_experiments.py
import typer
import subprocess
import sys
from loguru import logger

app = typer.Typer()

# --- ¡AQUÍ DEFINES TUS EXPERIMENTOS! ---
# Lista de diccionarios. Cada diccionario es un entrenamiento.
EXPERIMENTS = [
    {"model": "yolov8n", "epochs": 25},
    {"model": "yolov8n", "epochs": 150},
    {"model": "yolov8s", "epochs": 25},
    {"model": "yolov8s", "epochs": 150},
    {"model": "yolov8m", "epochs": 25},
    {"model": "yolov8m", "epochs": 150},
    # Puedes añadir más aquí...
    # {"model": "yolov8m", "epochs": 150},
]

# -----------------------------------------

@app.command()
def run():
    """
    Lanza secuencialmente todos los entrenamientos definidos en la lista EXPERIMENTS.
    """
    logger.info(f"🚀 Se van a lanzar {len(EXPERIMENTS)} experimentos de entrenamiento.")
    
    for i, exp in enumerate(EXPERIMENTS):
        model_name = exp["model"]
        num_epochs = exp["epochs"]
        
        logger.info("---------------------------------------------------------")
        logger.info(f"▶️  Iniciando Experimento {i+1}/{len(EXPERIMENTS)}: Modelo={model_name}, Épocas={num_epochs}")
        logger.info("---------------------------------------------------------")

        # Construir el comando que vamos a ejecutar en la terminal
        command = [
            sys.executable,  # Usa el mismo intérprete de Python que está ejecutando este script
            "src/modeling/train_ultralytics.py",
            "--model",
            model_name,
            "--epochs",
            str(num_epochs),
        ]

        try:
            # Ejecutar el comando del entrenamiento
            # Se usa Popen para ver la salida en tiempo real
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            
            # Imprimir la salida del subproceso en tiempo real
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            rc = process.poll() # Obtener el código de retorno
            
            if rc == 0:
                logger.success(f"✅ Experimento {i+1} completado con éxito.")
            else:
                logger.error(f"❌ El Experimento {i+1} falló con código de error {rc}.")

        except FileNotFoundError:
            logger.error(f"Error: No se encontró el script 'src/models/train.py'. Asegúrate de ejecutar este comando desde la raíz del proyecto.")
            break
        except Exception as e:
            logger.error(f"Ocurrió un error inesperado durante el experimento {i+1}: {e}")
            break

    logger.info("🏁 Todos los experimentos han finalizado.")


if __name__ == "__main__":
    app()