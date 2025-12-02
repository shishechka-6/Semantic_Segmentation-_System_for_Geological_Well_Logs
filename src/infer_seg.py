
import os
from glob import glob
import zipfile

import numpy as np
from PIL import Image

import torch
import segmentation_models_pytorch as smp


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ind_to_tone = ckpt["ind_to_tone"]
    return ind_to_tone


def tone_map_restore(pred_map, ind_to_tone):
    h, w = pred_map.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls_idx, tone in ind_to_tone.items():
        out[pred_map == cls_idx] = tone
    return out


def fill_small_gaps(pred, bg_class=0):
    h, w = pred.shape
    out = pred.copy()
    padded = np.pad(pred, pad_width=1, mode="edge")

    for i in range(h):
        for j in range(w):
            if pred[i, j] == bg_class:
                neigh = padded[i:i+3, j:j+3].ravel()
                vals = neigh[neigh != bg_class]
                if vals.size > 0:
                    out[i, j] = np.bincount(vals).argmax()

    return out


def predict_with_tta(model, arr_pad, device):
    model.eval()

    def _prep(a):
        x = torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return x.to(device)

    with torch.no_grad():
        x = _prep(arr_pad)  

        # 1) оригинал
        l1 = model(x)

        # 2) горизонтальный флип
        x_h = torch.flip(x, dims=[3])
        l2 = model(x_h)
        l2 = torch.flip(l2, dims=[3])

        # 3) вертикальный флип
        x_v = torch.flip(x, dims=[2])
        l3 = model(x_v)
        l3 = torch.flip(l3, dims=[2])

        # 4) оба флипа
        x_hv = torch.flip(x, dims=[2, 3])
        l4 = model(x_hv)
        l4 = torch.flip(l4, dims=[2, 3])

        logits = (l1 + l2 + l3 + l4) / 4.0  # [1, C, H, W]
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # пути относительно корня проекта
    test_paths = sorted(glob("data/test_images/*.png"))
    checkpoint_path = "work/weights/seg_best.pt"
    out_dir = "work/runs"
    zip_path = "predict_target.zip"

    if not test_paths:
        raise RuntimeError("В data/test_images не найдено ни одного .png файла.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Не найден чекпоинт: {checkpoint_path}")

    os.makedirs(out_dir, exist_ok=True)

    # модель (как при обучении)
    num_classes = 40
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    ).to(device)

    print("Loading checkpoint:", checkpoint_path)
    ind_to_tone = load_checkpoint(model, checkpoint_path, device)

    # инференс
    for img_path in test_paths:
        print("Predicting:", img_path)
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)  # (H, W, 3)
        H, W = arr.shape[:2]

        # паддинг до кратности 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            arr_pad = np.pad(
                arr,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            arr_pad = arr

        # предсказание с TTA
        pred_pad = predict_with_tta(model, arr_pad, device)  # (H_pad, W_pad)

        # убираем паддинг
        pred = pred_pad[:H, :W]

        # локально заполняем мелкие чёрные дырки (фон = класс 0)
        pred_filled = fill_small_gaps(pred, bg_class=0)

        # индексы классов -> оттенки серого
        out_arr = tone_map_restore(pred_filled, ind_to_tone)
        out_img = Image.fromarray(out_arr, mode="L")

        name = os.path.basename(img_path)
        out_img.save(os.path.join(out_dir, name))

    print("Готово, маски лежат в", out_dir)

    # собираем архив для отправки
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for png_path in sorted(glob(os.path.join(out_dir, "*.png"))):
            zf.write(png_path, arcname=os.path.basename(png_path))

    print("Архив с результатами создан:", zip_path)


if __name__ == "__main__":
    main()
