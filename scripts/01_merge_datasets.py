'''from pathlib import Path
import os
import shutil
import random
import time

# =============================
# PATHS (UPDATED)
# =============================
RAW_PATH = Path(r"data1/raw")
OUTPUT_PATH = Path(r"data1/processed/merged_train_test")

# =============================
# SETTINGS
# =============================
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20
SEED = 42
random.seed(SEED)

# =============================
# IMAGE EXTENSIONS
# =============================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

# =============================
# RAW FOLDER -> CLASS MAPPING
# =============================
RAW_FOLDER_MAP = {
    # Bacterial
    "Cellulitis Impetigo and other Bacterial Infections": "bacterial",

    # Fungal
    "Nail Fungus and other Nail Disease": "fungal",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",

    # Viral
    "Herpes HPV and other STDs Photos": "viral",
    "Warts Molluscum and other Viral Infections": "viral",
}

CLASSES = ["bacterial", "fungal", "viral"]
SPLITS = ["train", "test"]


# ==========================================================
# SAFER CLEAN: do NOT delete folders (OneDrive locks them)
# ==========================================================
def clear_only_files(folder: Path):
    """Deletes only files inside a folder (keeps folder structure)."""
    if not folder.exists():
        return

    for root, dirs, files in os.walk(str(folder)):
        for f in files:
            fp = Path(root) / f
            try:
                fp.unlink()
            except PermissionError:
                time.sleep(1)
                try:
                    fp.unlink()
                except:
                    pass


def make_output_structure():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)


def get_all_images_recursive(folder: Path):
    """Returns list of all image file paths inside folder (recursive)."""
    imgs = []
    if not folder.exists():
        return imgs

    for root, dirs, files in os.walk(str(folder)):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                imgs.append(Path(root) / file)

    return imgs


def safe_copy(src: Path, dst: Path):
    """Copies with OneDrive-safe retry."""
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except PermissionError:
        time.sleep(1)
        try:
            shutil.copy2(str(src), str(dst))
            return True
        except:
            return False


def copy_images(img_list, dst_folder: Path, prefix=""):
    dst_folder.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in img_list:
        base = img.stem
        ext = img.suffix

        new_name = f"{prefix}{base}{ext}"
        dst = dst_folder / new_name

        if dst.exists():
            dst = dst_folder / f"{prefix}{base}_{random.randint(10000,99999)}{ext}"

        if safe_copy(img, dst):
            copied += 1

    return copied


def main():
    print("📌 Preparing output folders...")
    make_output_structure()

    print("🧹 Clearing old files (safe mode for OneDrive)...")
    for split in SPLITS:
        for cls in CLASSES:
            clear_only_files(OUTPUT_PATH / split / cls)

    # Collect class-wise images
    class_images = {c: [] for c in CLASSES}

    print("\n📌 Collecting images from RAW folders...\n")

    for raw_folder, out_class in RAW_FOLDER_MAP.items():
        src_folder = RAW_PATH / raw_folder
        imgs = get_all_images_recursive(src_folder)

        if len(imgs) == 0:
            print(f"⚠️ No images found in: {src_folder.as_posix()}")
            continue

        class_images[out_class].extend(imgs)
        print(f"✅ {raw_folder} -> {out_class} | {len(imgs)} images")

    print("\n📌 Splitting into train/test and copying...\n")

    for cls in CLASSES:
        imgs = class_images[cls]
        if len(imgs) == 0:
            print(f"❌ No images found for class: {cls}")
            continue

        random.shuffle(imgs)

        train_count = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:train_count]
        test_imgs = imgs[train_count:]

        train_dst = OUTPUT_PATH / "train" / cls
        test_dst = OUTPUT_PATH / "test" / cls

        train_copied = copy_images(train_imgs, train_dst, prefix=f"{cls}_train_")
        test_copied = copy_images(test_imgs, test_dst, prefix=f"{cls}_test_")

        print(f"📌 {cls.upper()} -> train={train_copied}, test={test_copied}")

    print("\n🎉 DONE!")
    print("Merged dataset created at:", OUTPUT_PATH.as_posix())
    print("\nNext run:")
    print("python scripts/02_split_train_val.py")


if __name__ == "__main__":
    main()
'''

















'''
from pathlib import Path
import os
import shutil
import random
import time

# =============================
# PATHS
# =============================
RAW_PATH = Path(r"data1/raw")
OTHERS_PATH = RAW_PATH / "others"   # ✅ your folder from screenshot
OUTPUT_PATH = Path(r"data1/processed/merged_train_test")

# =============================
# SETTINGS
# =============================
TRAIN_RATIO = 0.80
SEED = 42
random.seed(SEED)

# =============================
# IMAGE EXTENSIONS
# =============================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

# =============================
# RAW FOLDER -> CLASS MAPPING
# (ONLY for infectious)
# =============================
RAW_FOLDER_MAP = {
    # Bacterial
    "Cellulitis Impetigo and other Bacterial Infections": "bacterial",

    # Fungal
    "Nail Fungus and other Nail Disease": "fungal",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",

    # Viral
    "Herpes HPV and other STDs Photos": "viral",
    "Warts Molluscum and other Viral Infections": "viral",
}

CLASSES = ["bacterial", "fungal", "viral", "other"]
SPLITS = ["train", "test"]

# These are folders in RAW_PATH that you want to completely ignore
EXCLUDE_FOLDERS = [
    "Skin Lesion Dataset",
    "skin-disease-dataset",
    "others",  # IMPORTANT: because we load it separately
]


# ==========================================================
# SAFER CLEAN: do NOT delete folders (OneDrive locks them)
# ==========================================================
def clear_only_files(folder: Path):
    """Deletes only files inside a folder (keeps folder structure)."""
    if not folder.exists():
        return

    for root, dirs, files in os.walk(str(folder)):
        for f in files:
            fp = Path(root) / f
            try:
                fp.unlink()
            except PermissionError:
                time.sleep(1)
                try:
                    fp.unlink()
                except:
                    pass


def make_output_structure():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)


def get_all_images_recursive(folder: Path):
    """Returns list of all image file paths inside folder (recursive)."""
    imgs = []
    if not folder.exists():
        return imgs

    for root, dirs, files in os.walk(str(folder)):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                imgs.append(Path(root) / file)

    return imgs


def safe_copy(src: Path, dst: Path):
    """Copies with OneDrive-safe retry."""
    try:
        shutil.copy2(str(src), str(dst))
        return True
    except PermissionError:
        time.sleep(1)
        try:
            shutil.copy2(str(src), str(dst))
            return True
        except:
            return False


def copy_images(img_list, dst_folder: Path, prefix=""):
    dst_folder.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in img_list:
        base = img.stem
        ext = img.suffix

        new_name = f"{prefix}{base}{ext}"
        dst = dst_folder / new_name

        if dst.exists():
            dst = dst_folder / f"{prefix}{base}_{random.randint(10000,99999)}{ext}"

        if safe_copy(img, dst):
            copied += 1

    return copied


def list_subfolders(folder: Path):
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.iterdir() if p.is_dir()])


def main():
    print("📌 Preparing output folders...")
    make_output_structure()

    print("🧹 Clearing old files (safe mode for OneDrive)...")
    for split in SPLITS:
        for cls in CLASSES:
            clear_only_files(OUTPUT_PATH / split / cls)

    # Collect class-wise images
    class_images = {c: [] for c in CLASSES}

    # =====================================================
    # PART 1: Collect infectious classes from RAW_PATH
    # =====================================================
    print("\n📌 Collecting Bacterial/Fungal/Viral from RAW folders...\n")

    raw_folders = list_subfolders(RAW_PATH)

    for raw_folder in raw_folders:
        if raw_folder in EXCLUDE_FOLDERS:
            print(f"⛔ Skipped (excluded): {raw_folder}")
            continue

        if raw_folder not in RAW_FOLDER_MAP:
            print(f"⛔ Skipped (not mapped): {raw_folder}")
            continue

        out_class = RAW_FOLDER_MAP[raw_folder]

        src_folder = RAW_PATH / raw_folder
        imgs = get_all_images_recursive(src_folder)

        if len(imgs) == 0:
            print(f"⚠️ No images found in: {src_folder.as_posix()}")
            continue

        class_images[out_class].extend(imgs)
        print(f"✅ {raw_folder} -> {out_class} | {len(imgs)} images")

    # =====================================================
    # PART 2: Collect OTHER from data1/raw/others/
    # =====================================================
    print("\n📌 Collecting OTHER class from:", OTHERS_PATH.as_posix())

    if not OTHERS_PATH.exists():
        print("\n❌ ERROR: others folder not found!")
        print("Expected:", OTHERS_PATH.as_posix())
        print("Create: data1/raw/others/ and add disease folders inside.")
        return

    other_folders = list_subfolders(OTHERS_PATH)

    if len(other_folders) == 0:
        print("\n❌ ERROR: No subfolders found inside:", OTHERS_PATH.as_posix())
        return

    for folder_name in other_folders:
        src_folder = OTHERS_PATH / folder_name
        imgs = get_all_images_recursive(src_folder)

        if len(imgs) == 0:
            print(f"⚠️ No images found in: {src_folder.as_posix()}")
            continue

        class_images["other"].extend(imgs)
        print(f"✅ {folder_name} -> other | {len(imgs)} images")

    # =====================================================
    # Split into train/test and copy
    # =====================================================
    print("\n📌 Splitting into train/test and copying...\n")

    for cls in CLASSES:
        imgs = class_images[cls]

        if len(imgs) == 0:
            print(f"❌ No images found for class: {cls}")
            continue

        random.shuffle(imgs)

        train_count = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:train_count]
        test_imgs = imgs[train_count:]

        train_dst = OUTPUT_PATH / "train" / cls
        test_dst = OUTPUT_PATH / "test" / cls

        train_copied = copy_images(train_imgs, train_dst, prefix=f"{cls}_train_")
        test_copied = copy_images(test_imgs, test_dst, prefix=f"{cls}_test_")

        print(f"📌 {cls.upper()} -> train={train_copied}, test={test_copied}")

    print("\n🎉 DONE!")
    print("Merged dataset created at:", OUTPUT_PATH.as_posix())
    print("\nNext run:")
    print("python scripts/02_split_train_val.py")


if __name__ == "__main__":
    main()
'''











'''
from pathlib import Path
import os
import shutil
import random
import time

# =============================
# PATHS
# =============================
RAW_PATH = Path(r"data1/raw")
OTHERS_PATH = RAW_PATH / "others"
OUTPUT_PATH = Path(r"data1/processed/merged_train_test")

# =============================
# SETTINGS
# =============================
TRAIN_RATIO = 0.80
SEED = 42
random.seed(SEED)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")

# =============================
# RAW FOLDER → CLASS MAP
# =============================
RAW_FOLDER_MAP = {

    # Bacterial
    "Cellulitis Impetigo and other Bacterial Infections": "bacterial",

    # Fungal
    "Nail Fungus and other Nail Disease": "fungal",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",

    # Viral
    "Herpes HPV and other STDs Photos": "viral",
    "Warts Molluscum and other Viral Infections": "viral",
}

CLASSES = ["bacterial", "fungal", "viral", "other"]
SPLITS = ["train", "test"]

EXCLUDE_FOLDERS = [
    "Skin Lesion Dataset",
    "skin-disease-dataset",
    "others"
]


def clear_only_files(folder: Path):
    if not folder.exists():
        return
    for root, dirs, files in os.walk(str(folder)):
        for f in files:
            try:
                (Path(root) / f).unlink()
            except:
                pass


def make_output_structure():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)


def get_all_images_recursive(folder: Path):
    imgs = []
    for root, dirs, files in os.walk(str(folder)):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                imgs.append(Path(root) / file)
    return imgs


def main():
    print("Preparing folders...")
    make_output_structure()

    for split in SPLITS:
        for cls in CLASSES:
            clear_only_files(OUTPUT_PATH / split / cls)

    class_images = {c: [] for c in CLASSES}

    # Infectious folders
    for raw_folder in RAW_PATH.iterdir():
        if not raw_folder.is_dir():
            continue

        name = raw_folder.name

        if name in EXCLUDE_FOLDERS:
            continue

        if name not in RAW_FOLDER_MAP:
            continue

        cls = RAW_FOLDER_MAP[name]
        imgs = get_all_images_recursive(raw_folder)
        class_images[cls].extend(imgs)

        print(f"{name} → {cls} | {len(imgs)} images")

    # Other category
    for folder in (OTHERS_PATH).iterdir():
        if folder.is_dir():
            imgs = get_all_images_recursive(folder)
            class_images["other"].extend(imgs)
            print(f"{folder.name} → other | {len(imgs)} images")

    # Split
    for cls in CLASSES:
        imgs = class_images[cls]
        random.shuffle(imgs)

        split_idx = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:split_idx]
        test_imgs = imgs[split_idx:]

        for img in train_imgs:
            shutil.copy2(img, OUTPUT_PATH / "train" / cls / img.name)

        for img in test_imgs:
            shutil.copy2(img, OUTPUT_PATH / "test" / cls / img.name)

        print(f"{cls}: train={len(train_imgs)} test={len(test_imgs)}")

    print("DONE!")


if __name__ == "__main__":
    main()'''







from pathlib import Path
import os
import shutil
import random

# =============================
# PATHS
# =============================
RAW_PATH = Path(r"data1/raw")
OTHERS_PATH = RAW_PATH / "others"

# EXTRA AUGMENTED BACTERIAL PATH
EXTRA_BACTERIAL_PATH = Path(r"data1/processed/final_4class/train/bacterial")

OUTPUT_PATH = Path(r"data1/processed/merged_train_test")

# =============================
# SETTINGS
# =============================
TRAIN_RATIO = 0.80
SEED = 42
random.seed(SEED)

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp",".jfif",".tif",".tiff")

RAW_FOLDER_MAP = {

    "Cellulitis Impetigo and other Bacterial Infections": "bacterial",

    "Nail Fungus and other Nail Disease": "fungal",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",

    "Herpes HPV and other STDs Photos": "viral",
    "Warts Molluscum and other Viral Infections": "viral",
}

CLASSES = ["bacterial","fungal","viral","other"]
SPLITS = ["train","test"]

EXCLUDE_FOLDERS = [
    "Skin Lesion Dataset",
    "skin-disease-dataset",
    "others"
]


def clear_only_files(folder: Path):

    if not folder.exists():
        return

    for root,dirs,files in os.walk(str(folder)):
        for f in files:
            try:
                (Path(root)/f).unlink()
            except:
                pass


def make_output_structure():

    OUTPUT_PATH.mkdir(parents=True,exist_ok=True)

    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_PATH/split/cls).mkdir(parents=True,exist_ok=True)


def get_all_images_recursive(folder: Path):

    imgs=[]

    if not folder.exists():
        return imgs

    for root,dirs,files in os.walk(str(folder)):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                imgs.append(Path(root)/file)

    return imgs


def main():

    print("Preparing folders...")
    make_output_structure()

    for split in SPLITS:
        for cls in CLASSES:
            clear_only_files(OUTPUT_PATH/split/cls)

    class_images={c:[] for c in CLASSES}

    # ==========================================
    # LOAD NORMAL DATASET
    # ==========================================
    for raw_folder in RAW_PATH.iterdir():

        if not raw_folder.is_dir():
            continue

        name=raw_folder.name

        if name in EXCLUDE_FOLDERS:
            continue

        if name not in RAW_FOLDER_MAP:
            continue

        cls=RAW_FOLDER_MAP[name]

        imgs=get_all_images_recursive(raw_folder)

        class_images[cls].extend(imgs)

        print(f"{name} → {cls} | {len(imgs)} images")

    # ==========================================
    # ADD AUGMENTED BACTERIAL IMAGES
    # ==========================================
    extra_bacterial = get_all_images_recursive(EXTRA_BACTERIAL_PATH)

    class_images["bacterial"].extend(extra_bacterial)

    print(f"Augmented bacterial images added → {len(extra_bacterial)}")

    # ==========================================
    # OTHER CLASS
    # ==========================================
    for folder in (OTHERS_PATH).iterdir():

        if folder.is_dir():

            imgs=get_all_images_recursive(folder)

            class_images["other"].extend(imgs)

            print(f"{folder.name} → other | {len(imgs)} images")

    # ==========================================
    # SPLIT TRAIN / TEST
    # ==========================================
    for cls in CLASSES:

        imgs=class_images[cls]

        random.shuffle(imgs)

        split_idx=int(len(imgs)*TRAIN_RATIO)

        train_imgs=imgs[:split_idx]
        test_imgs=imgs[split_idx:]

        for img in train_imgs:
            shutil.copy2(img,OUTPUT_PATH/"train"/cls/img.name)

        for img in test_imgs:
            shutil.copy2(img,OUTPUT_PATH/"test"/cls/img.name)

        print(f"{cls}: train={len(train_imgs)} test={len(test_imgs)}")

    print("\nDONE!")


if __name__=="__main__":
    main()