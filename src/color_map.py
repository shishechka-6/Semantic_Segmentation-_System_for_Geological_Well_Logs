from PIL import Image
import numpy as np

def build_class_maps(mask_paths):
    tone = set()

    for p in mask_paths:
        arr = np.array(Image.open(p))
        tone.update(np.unique(arr).tolist())

    tone = sorted(tone)

    tone_to_ind = {}
    for index, value in enumerate(tone):
        tone_to_ind[value] = index

    ind_to_tone = {}
    for value, index in tone_to_ind.items():
        ind_to_tone[index] = value

    return tone_to_ind, ind_to_tone
