import os
import numpy as np
from PIL import Image

# Cityscapes official id → trainId mapping table
id2trainId = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1,
    9: 255, 10: 255,
    11: 2, 12: 3, 13: 4,
    14: 255, 15: 255, 16: 255,
    17: 5, 18: 255,
    19: 6, 20: 7,
    21: 8, 22: 9,
    23: 10,
    24: 11, 25: 12,
    26: 13, 27: 14, 28: 15,
    29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
    -1: 255
}


def convert_labelIds_to_trainIds(labelId_path, save_path):
    """Convert single image"""
    labelId = np.array(Image.open(labelId_path), dtype=np.int32)
    trainId = np.ones_like(labelId, dtype=np.uint8) * 255  # Default ignore

    for lid, tid in id2trainId.items():
        trainId[labelId == lid] = tid

    Image.fromarray(trainId).save(save_path)


def batch_convert(root_input, root_output):
    """Process the entire root directory in batches, maintaining the 6 subfolder structure"""
    if not os.path.exists(root_output):
        os.makedirs(root_output)

    # Traverse 6 subfolders
    cities = [d for d in os.listdir(root_input) if os.path.isdir(os.path.join(root_input, d))]
    print(f"Found {len(cities)} city folders: {cities}")

    for city in cities:
        city_dir = os.path.join(root_input, city)
        save_city_dir = os.path.join(root_output, city)
        os.makedirs(save_city_dir, exist_ok=True)

        # Traverse all images in this city
        files = [f for f in os.listdir(city_dir) if f.endswith(".png")]
        print(f"{city}: Found {len(files)} images, starting conversion...")

        for fname in files:
            src_path = os.path.join(city_dir, fname)
            dst_path = os.path.join(save_city_dir, fname.replace("labelIds", "labelTrainIds"))
            convert_labelIds_to_trainIds(src_path, dst_path)

    print(f"\n✅ All cities converted successfully, results saved in: {root_output}")


if __name__ == "__main__":
    root_input = r"E:\HCB Downloads\LightNet-master\LightNet-master\datasets\mobilenetv2plus"  # Original root directory
    root_output = r"E:\HCB Downloads\LightNet-master\LightNet-master\datasets\mobilenetv2plus_trainIds"  # Output root directory

    batch_convert(root_input, root_output)
