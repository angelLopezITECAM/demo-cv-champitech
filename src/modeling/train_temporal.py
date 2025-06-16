import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos




class LoadImagesSlidingWindow(LoadImagesAndVideos):
    """
    DataLoader that returns a sliding window of images for temporal context.
    Each sample is a stack of `window_size` RGB images along the channel axis, but labels
    correspond only to the central frame.
    """
    def __init__(self, *args, window_size=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = window_size
        # Build index sequences over all images (assumed sorted by time)
        N = len(self.img_files)
        self.seqs = [tuple(range(i, i + self.ws)) for i in range(N - self.ws + 1)]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        indices = self.seqs[idx]
        ims, labels, paths, shapes = [], None, [], None
        # Iterate through the window
        for j in indices:
            im, lab, path, shape = super().__getitem__(j)
            ims.append(im)
            # Only keep labels of the central frame
            center = indices[len(indices) // 2]
            if j == center:
                labels = lab
            paths.append(path)
            shapes = shape
        # Stack along channel dimension: [3*ws, H, W]
        im_seq = torch.cat(ims, dim=0)
        # Return sequence, central labels, path of central frame, and shapes
        center_path = paths[len(paths) // 2]
        return im_seq, labels, center_path, shapes

if __name__ == '__main__':
    # Configuration
    data_yaml = 'data.yaml'            # Ruta a tu config de datos
    window_size = 3                    # Tamaño de la ventana deslizante
    epochs = 50
    imgsz = 640

    # 1) Instanciar el modelo YOLOv8
    model = YOLO('yolov8s.pt', ch=3 * window_size)  # ch=3*window_size para aceptar secuencias RGB

    # 2) Adaptar la primera convolución para aceptar 3*window_size canales
    old_conv = model.model[0].conv.conv
    new_conv = torch.nn.Conv2d(
        in_channels=3 * window_size,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )
    # Inicializar pesos: copia los pesos RGB al canal central
    with torch.no_grad():
        # old_conv.weight shape: [out,3,k,k]
        new_conv.weight.zero_()
        # canales central (offset ws//2)
        start = window_size // 2 * 3
        new_conv.weight[:, start:start+3] = old_conv.weight
    model.model[0].conv.conv = new_conv

    # 3) Entrenar con loader personalizado
    #    Pasamos el DataLoader directamente a model.train
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        loader=LoadImagesSlidingWindow,
        loader_args={'window_size': window_size}
    )
