import torch
from glob import glob
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from color_map import build_class_maps
from dataset import SegDataset


def main():
    # 0. Устройство (CPU, потому что CUDA нет)
    device = torch.device("cpu")

    # 1. Пути к данным
    img_paths = sorted(glob("data/train_images/*.png"))
    msk_paths = sorted(glob("data/train_masks/*.png"))
    print("Картинок:", len(img_paths))

    # 2. Строим tone -> index
    tone_to_ind, ind_to_tone = build_class_maps(msk_paths)
    num_classes = len(tone_to_ind)
    print("Классов:", num_classes)

    # 3. Датасет и даталоудер (просто для проверки)
    ds = SegDataset(img_paths, msk_paths, tone_to_ind, is_train=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False)

    # 4. Модель DeepLabV3+
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # 5. Берём один батч
    images, masks = next(iter(dl))
    images = images.to(device)
    masks = masks.to(device)

    print("batch images:", images.shape)  # [B, 3, 640, 640]
    print("batch masks:", masks.shape)    # [B, 640, 640]

    # 6. Прогон через модель
    logits = model(images)  # [B, C, H, W]
    print("logits shape:", logits.shape)

    # 7. Простой лосс (кросс-энтропия)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, masks)
    print("loss:", loss.item())


if __name__ == "__main__":
    main()
