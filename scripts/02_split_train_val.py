'''import os
import shutil
import random
import time

INPUT_PATH = r"data1/processed/merged_train_test"
OUTPUT_PATH = r"data1/processed/final_3class"

VAL_RATIO = 0.20
SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")
CLASSES = ["bacterial", "fungal", "viral"]


def clear_only_files(folder):
    """OneDrive safe cleaning."""
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return

    for root, dirs, files in os.walk(folder):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except PermissionError:
                time.sleep(1)
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass


def list_images(folder):
    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]


def safe_copy(src, dst):
    try:
        shutil.copy2(src, dst)
        return True
    except PermissionError:
        time.sleep(1)
        try:
            shutil.copy2(src, dst)
            return True
        except:
            return False


def copy_files(file_list, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    copied = 0

    for f in file_list:
        src = os.path.join(src_folder, f)
        dst = os.path.join(dst_folder, f)

        if safe_copy(src, dst):
            copied += 1

    return copied


def main():
    random.seed(SEED)

    print("📌 Preparing output folder structure...")

    # Create structure
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

    print("🧹 Clearing old files (safe mode)...")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            clear_only_files(os.path.join(OUTPUT_PATH, split, cls))

    # ============================
    # Split train -> train + val
    # ============================
    print("\n📌 Splitting training data into train + val...\n")

    for cls in CLASSES:
        src_folder = os.path.join(INPUT_PATH, "train", cls)
        images = list_images(src_folder)
        random.shuffle(images)

        val_count = int(len(images) * VAL_RATIO)
        val_imgs = images[:val_count]
        train_imgs = images[val_count:]

        train_copied = copy_files(train_imgs, src_folder, os.path.join(OUTPUT_PATH, "train", cls))
        val_copied = copy_files(val_imgs, src_folder, os.path.join(OUTPUT_PATH, "val", cls))

        print(f"✅ {cls}: train={train_copied} val={val_copied}")

    # ============================
    # Copy test as it is
    # ============================
    print("\n📌 Copying test set...\n")
    for cls in CLASSES:
        src = os.path.join(INPUT_PATH, "test", cls)
        dst = os.path.join(OUTPUT_PATH, "test", cls)

        if os.path.exists(src):
            for f in list_images(src):
                safe_copy(os.path.join(src, f), os.path.join(dst, f))

        print(f"✅ Copied test/{cls}")

    print("\n🎉 DONE!")
    print("Final dataset ready at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
'''






'''import os
import shutil
import random
import time

# =====================================================
# INPUT
# =====================================================
INPUT_PATH = r"data1/processed/merged_train_test"

# =====================================================
# OUTPUT
# =====================================================
OUTPUT_PATH = r"data1/processed/final_4class"

VAL_RATIO = 0.20
SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

# 🔥 4 classes now
CLASSES = ["bacterial", "fungal", "viral", "other"]


# =====================================================
# UTILS
# =====================================================
def clear_only_files(folder):
    """OneDrive safe cleaning."""
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return

    for root, dirs, files in os.walk(folder):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except PermissionError:
                time.sleep(1)
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass


def list_images(folder):
    if not os.path.exists(folder):
        return []

    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]


def safe_copy(src, dst):
    try:
        shutil.copy2(src, dst)
        return True
    except PermissionError:
        time.sleep(1)
        try:
            shutil.copy2(src, dst)
            return True
        except:
            return False


def copy_files(file_list, src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    copied = 0

    for f in file_list:
        src = os.path.join(src_folder, f)
        dst = os.path.join(dst_folder, f)

        if safe_copy(src, dst):
            copied += 1

    return copied


# =====================================================
# MAIN
# =====================================================
def main():
    random.seed(SEED)

    print("📌 Preparing output folder structure...")

    # Create structure
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

    print("🧹 Clearing old files (safe mode)...")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            clear_only_files(os.path.join(OUTPUT_PATH, split, cls))

    # =====================================================
    # 1) Split train -> train + val (ALL 4 CLASSES)
    # =====================================================
    print("\n📌 Splitting training data into train + val...\n")

    for cls in CLASSES:
        src_folder = os.path.join(INPUT_PATH, "train", cls)

        images = list_images(src_folder)
        random.shuffle(images)

        if len(images) == 0:
            print(f"❌ No images found in train/{cls}")
            continue

        val_count = int(len(images) * VAL_RATIO)
        val_imgs = images[:val_count]
        train_imgs = images[val_count:]

        train_copied = copy_files(train_imgs, src_folder, os.path.join(OUTPUT_PATH, "train", cls))
        val_copied = copy_files(val_imgs, src_folder, os.path.join(OUTPUT_PATH, "val", cls))

        print(f"✅ {cls}: train={train_copied} val={val_copied}")

    # =====================================================
    # 2) Copy test as it is (ALL 4 CLASSES)
    # =====================================================
    print("\n📌 Copying test set...\n")

    for cls in CLASSES:
        src = os.path.join(INPUT_PATH, "test", cls)
        dst = os.path.join(OUTPUT_PATH, "test", cls)

        if not os.path.exists(src):
            print(f"❌ Missing folder: test/{cls}")
            continue

        for f in list_images(src):
            safe_copy(os.path.join(src, f), os.path.join(dst, f))

        print(f"✅ Copied test/{cls}")

    # =====================================================
    # DONE
    # =====================================================
    print("\n🎉 DONE!")
    print("Final dataset ready at:", OUTPUT_PATH)
    print("\nNow train using:")
    print(r'DATASET_PATH = r"data1/processed/final_4class"')


if __name__ == "__main__":
    main()
'''





import os
import shutil
import random
import time

# ====================================
# INPUT FROM MERGE SCRIPT
# ====================================
INPUT_PATH = r"data1/processed/merged_train_test"

# ====================================
# FINAL DATASET FOR TRAINING
# ====================================
OUTPUT_PATH = r"data1/processed/final_4class"

VAL_RATIO = 0.20
SEED = 42

CLASSES = ["bacterial", "fungal", "viral", "other"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

random.seed(SEED)


# ====================================
# SAFE FILE CLEAN (ONEDRIVE SAFE)
# ====================================
def clear_only_files(folder):

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return

    for root, dirs, files in os.walk(folder):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except PermissionError:
                time.sleep(1)
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass


# ====================================
# LIST IMAGES
# ====================================
def list_images(folder):

    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]


# ====================================
# SAFE COPY
# ====================================
def safe_copy(src, dst):

    try:
        shutil.copy2(src, dst)
        return True
    except PermissionError:
        time.sleep(1)
        try:
            shutil.copy2(src, dst)
            return True
        except:
            return False


# ====================================
# COPY FILES
# ====================================
def copy_files(file_list, src_folder, dst_folder):

    os.makedirs(dst_folder, exist_ok=True)

    copied = 0

    for f in file_list:

        src = os.path.join(src_folder, f)
        dst = os.path.join(dst_folder, f)

        if safe_copy(src, dst):
            copied += 1

    return copied


# ====================================
# MAIN
# ====================================
def main():

    print("Preparing output folder structure...")

    # Create folder structure
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_PATH, split, cls), exist_ok=True)

    print("Clearing old files...")

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            clear_only_files(os.path.join(OUTPUT_PATH, split, cls))

    # ====================================
    # SPLIT TRAIN → TRAIN + VAL
    # ====================================
    print("\nSplitting train into train + val\n")

    for cls in CLASSES:

        src_folder = os.path.join(INPUT_PATH, "train", cls)

        images = list_images(src_folder)

        if len(images) == 0:
            print(f"No images found for {cls}")
            continue

        random.shuffle(images)

        val_count = int(len(images) * VAL_RATIO)

        val_imgs = images[:val_count]
        train_imgs = images[val_count:]

        train_copied = copy_files(
            train_imgs,
            src_folder,
            os.path.join(OUTPUT_PATH, "train", cls)
        )

        val_copied = copy_files(
            val_imgs,
            src_folder,
            os.path.join(OUTPUT_PATH, "val", cls)
        )

        print(f"{cls}: train={train_copied} val={val_copied}")

    # ====================================
    # COPY TEST SET
    # ====================================
    print("\nCopying test set...\n")

    for cls in CLASSES:

        src = os.path.join(INPUT_PATH, "test", cls)
        dst = os.path.join(OUTPUT_PATH, "test", cls)

        if not os.path.exists(src):
            print(f"Missing test folder: {cls}")
            continue

        for f in list_images(src):
            safe_copy(os.path.join(src, f), os.path.join(dst, f))

        print(f"Copied test/{cls}")

    print("\nDONE!")
    print("Final dataset ready at:", OUTPUT_PATH)


if __name__ == "__main__":
    main()