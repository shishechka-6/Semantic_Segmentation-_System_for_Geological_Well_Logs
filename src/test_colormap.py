from glob import glob
from color_map import build_class_maps

def main():
    mask_paths = sorted(glob("data/train_masks/*.png"))
    if not mask_paths:
        print("Не найдены маски в data/train_masks/*.png")
        return

    tone_to_ind, ind_to_tone = build_class_maps(mask_paths)

    print("Всего уникальных значений (классов):", len(tone_to_ind))
    print("\nПервые несколько value -> index:")
    for tone in list(sorted(tone_to_ind.keys()))[:10]:
        print(f"  tone {tone:3} -> class_index {tone_to_ind[tone]}")

    print("\nПервые несколько index -> value:")
    for i in range(min(10, len(ind_to_tone))):
        print(f"  class_index {i:2} -> tone {ind_to_tone[i]}")

if __name__ == "__main__":
    main()
