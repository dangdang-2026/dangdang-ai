import numpy as np
from PIL import Image, ImageOps

RESIZE = 256  # eval_tf: Resize(256)
CROP = 224  # eval_tf: CenterCrop(224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # training/train.py 와 같은 숫자
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")  # 폰 사진 회전 바로잡기 → 색 3개

    width, height = image.size
    if width <= height:  # 짧은 쪽을 256 으로
        size = (RESIZE, int(RESIZE * height / width))
    else:
        size = (int(RESIZE * width / height), RESIZE)
    image = image.resize(size, Image.Resampling.BILINEAR)

    left = int(round((image.width - CROP) / 2.0))  # 가운데 224×224
    top = int(round((image.height - CROP) / 2.0))
    image = image.crop((left, top, left + CROP, top + CROP))

    array = np.asarray(image, dtype=np.float32) / 255.0  # 0~255 → 0~1
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1)[np.newaxis]  # (세로, 가로, 색) → (1, 색, 세로, 가로)
