# Este código iría en tu script de entrenamiento o en un fichero aparte como src/data/datasets.py
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class CustomYOLODataset(Dataset):
    """
    Dataset personalizado que lee un fichero CSV para cargar imágenes y etiquetas.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Construye las rutas completas a los ficheros usando el CSV
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index]['image_path'])
        label_path = os.path.join(self.root_dir, self.annotations.iloc[index]['label_path'])
        
        # Carga la imagen
        image = Image.open(img_path).convert("RGB")
        
        # Carga las etiquetas del fichero .txt de YOLO
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                class_id, x, y, w, h = [float(x) for x in line.strip().split()]
                boxes.append([class_id, x, y, w, h])
        
        boxes = torch.tensor(boxes)
        target = {"boxes": boxes}

        # Aplica transformaciones si existen
        if self.transform:
            image = self.transform(image)
            
        return image, target