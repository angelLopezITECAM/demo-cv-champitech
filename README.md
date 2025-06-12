
# demo-cv-champitech
  

## Crear dataset

  

La función de este comando es preparar los datos para el modelo. Para ello, genera dos grupos de imágenes a partir de la carpeta data/raw: uno de entrenamiento con el 80% de los datos y otro de validación con el 20% restante, asegurando que la selección sea aleatoria.

```
python src/dataset_ultralytics.py
```

  

## Entrenamiento
Para entrenar el modelo se usa YOLO, en cualquiera de sus versiones y modelos.
Entre los modelos que hay se encuentran: 
- YOLOv3
- YOLOv4
- YOLOv5
- YOLOv6
- YOLOv7
- YOLOv8
- YOLOv9
- YOLOv10
- YOLO11
- YOLO12

Los modelos disponibles son: 
| Nombre	      | Precisión | Velocidad  | Recursos necesarios |
|-----------------|-----------|------------|---------------------|
| Nano (n)        | Más baja  | Más rápida | Muy bajos           |
| Small (s)       | Baja      | Rápida     | Bajos               |
| Medium (m)      | Media     | Media      | Medios              |
| Large (l)       | Alta      | Lenta      | Altos               |
| Extra large (x) | Más alta  | Más lenta  | Muy altos           |

El comando para lanzar un entrenamiento es el siguiente:

```
python src/modeling/train_ultralytics.py --model yolo12n --epochs 50
```
 Dónde *model*, es el modelo a utilizar, y *epochs* es el número de veces que el modelo va a mirarse el dataset de principio a fin


