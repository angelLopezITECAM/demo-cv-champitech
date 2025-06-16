
# demo-cv-champitech

## Estructura carpeta data

- data
    - external
    - interim
    - processed
    - raw
        - primordia
            - *EXTRAER EL ZIP DEL DATASET DE LA WEB

## Crear dataset

La función de este comando es preparar los datos para el modelo. Para ello, genera dos grupos de imágenes a partir de la carpeta data/raw: uno de entrenamiento con el 80% de los datos y otro de validación con el 20% restante, asegurando que la selección sea aleatoria.

```bash
python src/dataset_ultralytics.py
```

## Entrenamiento
Para entrenar el modelo se usa la arquitectura **YOLO**, implementada a través de la librería `Ultralytics`. Esta librería da acceso a varias versiones del modelo.

Cada versión dispone de diferentes tamaños, que permiten equilibrar precisión y velocidad:

| Nombre      | Identificador | Precisión | Velocidad  | Recursos Necesarios |
| :---------- | :------------ | :-------- | :--------- | :------------------ |
| Nano        | `n`           | Más Baja  | Más Rápida | Muy Bajos           |
| Small       | `s`           | Baja      | Rápida     | Bajos               |
| Medium      | `m`           | Media     | Media      | Medios              |
| Large       | `l`           | Alta      | Lenta      | Altos               |
| Extra Large | `x`           | Más Alta  | Más Lenta  | Muy Altos           |

El comando para lanzar un entrenamiento es el siguiente:

```bash
python src/modeling/train_ultralytics.py --model yolov8m --epochs 50
```

Donde:
- `--model`: Es el modelo a utilizar (ej. `yolov10n` combina la versión 10 con el tamaño nano).
- `--epochs`: Es el número de veces que el modelo procesará el dataset completo.

## Entrenamiento INCREMENTAL

Para iniciar un entrenamiento de manera incremental, entrena con el día 1, valida con el 2, entrena con el día 1 y 2, valida con el 3...

Este comando hace el split de los dias para cada entreno

```bash
python src/modeling/train_incremental.py
```

### Lanzar experimentos para un dataset

Dentro del archivo `scripts/run_experiments`, están los entrenos con el modelo y los epochs

```bash
python scripts/run_experiments.py
```