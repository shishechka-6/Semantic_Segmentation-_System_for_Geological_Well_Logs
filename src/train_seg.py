
import os
import random
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from color_map import build_class_maps
from dataset import SegDataset


def split_train_val(image_paths, mask_paths, val_ratio=0.1, seed=42):
    indices = list(range(len(image_paths)))
    random.Random(seed).shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    def pick(idxs, arr):
        return [arr[i] for i in idxs]

    train_images = pick(train_idx, image_paths)
    train_masks = pick(train_idx, mask_paths)
    val_images = pick(val_idx, image_paths)
    val_masks = pick(val_idx, mask_paths)

    return train_images, train_masks, val_images, val_masks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Путь к данным
    img_paths = sorted(glob("data/train_images/*.png"))
    msk_paths = sorted(glob("data/train_masks/*.png"))
    print("Картинок:", len(img_paths))

    assert len(img_paths) == len(msk_paths) and len(img_paths) > 0, "Проблема с train данными"

    # Маппинги тон - класс
    tone_to_ind, ind_to_tone = build_class_maps(msk_paths)
    num_classes = len(tone_to_ind)
    print("Классов:", num_classes)

    # Деление на train / val
    train_images, train_masks, val_images, val_masks = split_train_val(
        img_paths, msk_paths, val_ratio=0.1, seed=42
    )
    print("Train:", len(train_images), "Val:", len(val_images))

    # Датасет и даталоудер
    train_ds = SegDataset(train_images, train_masks, tone_to_ind, is_train=True)
    val_ds = SegDataset(val_images, val_masks, tone_to_ind, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # 5. Модель
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # Лосс и оптимизатор
    ce_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    os.makedirs("work/weights", exist_ok=True)

    NUM_EPOCHS = 3
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Эпоха {epoch} ===")

    
        model.train()
        train_loss_sum = 0.0

        for images, masks in tqdm(train_loader, desc="train"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)               
            loss = ce_loss(logits, masks)       
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))


        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="val"):
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = ce_loss(logits, masks)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))

        print(f"train_loss = {avg_train_loss:.4f}   val_loss = {avg_val_loss:.4f}")

        #Сохранение чекпоинтов
        checkpoint = {
            "model": model.state_dict(),
            "tone_to_ind": tone_to_ind,
            "ind_to_tone": ind_to_tone,
        }
        torch.save(checkpoint, f"work/weights/seg_epoch{epoch:02d}_valloss{avg_val_loss:.4f}.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, "work/weights/seg_best.pt")
            print("Обновили лучшую модель")

    print("Обучение завершено. Лучшая вал. потеря:", best_val_loss)


if __name__ == "__main__":
    main()
