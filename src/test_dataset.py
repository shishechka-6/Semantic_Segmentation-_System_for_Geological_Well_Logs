from glob import glob
from dataset import SegDataset
from color_map import build_class_maps
import torch

def main():
    # 1. Собираем пути к картинкам и маскам
    image_paths = sorted(glob("data/train_images/*.png"))
    mask_paths = sorted(glob("data/train_masks/*.png"))
    print("Нашли:", len(image_paths), "картинок")

    # 2. Строим словарь оттенки -> индексы
    tone_to_ind, ind_to_tone = build_class_maps(mask_paths)

    # 3. Создаём датасет
    ds = SegDataset(image_paths, mask_paths, tone_to_ind, is_train=False)

    # 4. Берём один элемент
    image, mask = ds[0]

    print("Тип image:", type(image))
    print("shape image:", image.shape)   # ожидаем [3, H, W]
    print("Тип mask:", type(mask))
    print("shape mask:", mask.shape)     # [H, W]
    print("Уникальные значения mask:", torch.unique(mask))

if __name__ == "__main__":
    main()
