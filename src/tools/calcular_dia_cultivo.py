from datetime import datetime
import json
import os

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR

def calcular_dia_cultivo_desde_json(ruta_fichero_json: str) -> int:
    """
    Calcula el día de cultivo leyendo la fecha de inicio y la fecha de la toma
    directamente del fichero JSON.
    """
    with open(ruta_fichero_json, 'r') as f:
        metadata = json.load(f)

    # 1. Extraer y convertir ambas fechas
    # Se usan las claves que tú mismo has identificado
    fecha_inicio_str = metadata["dia_entrada"]
    fecha_imagen_str = metadata["fecha"]
    
    # 2. Convertir los strings a objetos datetime
    fecha_inicio = datetime.strptime(fecha_inicio_str, "%Y-%m-%d")
    fecha_imagen = datetime.strptime(fecha_imagen_str, "%Y-%m-%d")

    # 3. Calcular la diferencia y devolver el número de días
    diferencia = fecha_imagen - fecha_inicio
    return diferencia.days