import os

# ======================================
# DATASET LOCATION
# ======================================
DATASET_PATH = r"data1/processed/final_4class"

CLASSES = ["bacterial", "fungal", "viral", "other"]
SPLITS = ["train", "val", "test"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")


# ======================================
# COUNT IMAGES
# ======================================
def count_images(folder):

    if not os.path.exists(folder):
        return 0

    total = 0

    for root, dirs, files in os.walk(folder):

        for f in files:

            if f.lower().endswith(IMG_EXTS):
                total += 1

    return total


# ======================================
# MAIN
# ======================================
def main():

    print("\nDATASET REPORT")
    print("=" * 55)

    grand_total = 0

    for split in SPLITS:

        print(f"\n{split.upper()}")

        split_total = 0

        for cls in CLASSES:

            folder = os.path.join(DATASET_PATH, split, cls)

            c = count_images(folder)

            split_total += c

            print(f"{cls:10s}: {c}")

        grand_total += split_total

        print(f"{'TOTAL':10s}: {split_total}")

    print("\n" + "=" * 55)
    print("GRAND TOTAL:", grand_total)


if __name__ == "__main__":
    main()