import os
import random
from PIL import Image, ImageEnhance

DATASET = r"data1/processed/final_4class/train/bacterial"
TARGET_COUNT = 2300

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def augment(img):

    # random rotation
    angle = random.randint(-30,30)
    img = img.rotate(angle)

    # brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7,1.3))

    # contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7,1.3))

    # flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img


def main():

    images = list_images(DATASET)
    current = len(images)

    print("Current bacterial images:", current)

    if current >= TARGET_COUNT:
        print("Already balanced.")
        return

    needed = TARGET_COUNT - current
    print("Generating", needed, "augmented images")

    for i in range(needed):

        img_name = random.choice(images)
        img_path = os.path.join(DATASET, img_name)

        img = Image.open(img_path).convert("RGB")

        aug = augment(img)

        new_name = f"aug_{i}_{img_name}"
        aug.save(os.path.join(DATASET,new_name))

    print("Augmentation complete!")

if __name__ == "__main__":
    main()
