'''#EFFICIENTNETB0

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


DATASET_PATH = r"data1/processed/final_3class"

MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_plots(history, results_path):
    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, "accuracy_curve.png"))
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, "loss_curve.png"))
    plt.close()


def main():
    print("Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Save class names
    with open(os.path.join(MODELS_PATH, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # =========================
    # Model
    # =========================
    print("\nBuilding model...")

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =========================
    # Save model summary (FIXED)
    # =========================
    summary_path = os.path.join(RESULTS_PATH, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    print("Saved model summary:", summary_path)

    # =========================
    # Callbacks
    # =========================
    ckpt_path = os.path.join(MODELS_PATH, "best_model.keras")

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=3, factor=0.2, monitor="val_loss"),
        CSVLogger(os.path.join(RESULTS_PATH, "training_history.csv"))
    ]

    # Save run config
    run_config = {
        "dataset_path": DATASET_PATH,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "base_model": "EfficientNetB0",
        "optimizer": "Adam",
        "learning_rate": 1e-3
    }
    with open(os.path.join(RESULTS_PATH, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # =========================
    # Train
    # =========================
    print("\nTraining started...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save plots
    save_training_plots(history, RESULTS_PATH)
    print("Saved training plots: accuracy_curve.png and loss_curve.png")

    # Best epoch
    best_epoch = int(np.argmax(history.history["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history.history["val_accuracy"]))

    # =========================
    # Evaluate
    # =========================
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Accuracy:", test_acc)

    # =========================
    # Predictions + Confusion Matrix
    # =========================
    print("\nGenerating confusion matrix...")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix
    save_confusion_matrix(cm, class_names, os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    np.save(os.path.join(RESULTS_PATH, "confusion_matrix.npy"), cm)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(RESULTS_PATH, "confusion_matrix.csv"), index=True)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "classification_report": report
    }

    with open(os.path.join(RESULTS_PATH, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save final model
    model.save(os.path.join(MODELS_PATH, "final_model.keras"))

    print("\nDONE!")
    print("Saved:")
    print("Best model:", ckpt_path)
    print("Final model:", os.path.join(MODELS_PATH, "final_model.keras"))
    print("Results in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
'''








'''#EFFICIENTNETB3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


# =====================================================
# SETTINGS
# =====================================================
DATASET_PATH = r"data1/processed/final_3class"

MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
SEED = 42

# 🔥 Change this to try other models:
BACKBONE = "EfficientNetB3"
# Options:
# "EfficientNetB0"
# "EfficientNetB3"
# "EfficientNetV2S"
# "ResNet50V2"
# "DenseNet121"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# =====================================================
# BACKBONE SELECTOR
# =====================================================
def get_backbone(name, img_size):
    if name == "EfficientNetB0":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3)
        )

    elif name == "EfficientNetB3":
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        base = tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3)
        )

    elif name == "EfficientNetV2S":
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
        base = tf.keras.applications.EfficientNetV2S(
            include_top=False, weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3)
        )

    elif name == "ResNet50V2":
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3)
        )

    elif name == "DenseNet121":
        preprocess = tf.keras.applications.densenet.preprocess_input
        base = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet",
            input_shape=(img_size[0], img_size[1], 3)
        )

    else:
        raise ValueError(f"Unknown BACKBONE: {name}")

    return base, preprocess


# =====================================================
# UTILS
# =====================================================
def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_plots(history_df, results_path):
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, "accuracy_curve.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, "loss_curve.png"))
    plt.close()


# =====================================================
# MAIN
# =====================================================
def main():
    print("Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    print("Using backbone:", BACKBONE)

    # Save class names
    with open(os.path.join(MODELS_PATH, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # =====================================================
    # MODEL
    # =====================================================
    base_model, preprocess_fn = get_backbone(BACKBONE, IMG_SIZE)
    base_model.trainable = False

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.20),
        layers.RandomContrast(0.20),
    ])

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)
    x = preprocess_fn(x)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Save model summary
    summary_path = os.path.join(RESULTS_PATH, "model_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # =====================================================
    # CALLBACKS
    # =====================================================
    ckpt_path = os.path.join(MODELS_PATH, f"best_model_{BACKBONE}.keras")

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=3, factor=0.2, monitor="val_loss"),
        CSVLogger(os.path.join(RESULTS_PATH, f"training_history_{BACKBONE}.csv"))
    ]

    # =====================================================
    # STAGE 1 TRAINING (Frozen)
    # =====================================================
    print("\nStage 1 Training (Frozen Backbone)...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks
    )

    # =====================================================
    # STAGE 2 TRAINING (Fine-tuning)
    # =====================================================
    print("\nStage 2 Training (Fine-tuning last layers)...")

    base_model.trainable = True

    # Unfreeze only last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks
    )

    # =====================================================
    # COMBINE HISTORY
    # =====================================================
    hist1 = pd.DataFrame(history1.history)
    hist2 = pd.DataFrame(history2.history)

    hist1["stage"] = "stage1"
    hist2["stage"] = "stage2"

    history_df = pd.concat([hist1, hist2], ignore_index=True)
    history_df.to_csv(os.path.join(RESULTS_PATH, f"full_history_{BACKBONE}.csv"), index=False)

    save_training_plots(history_df, RESULTS_PATH)

    # Best epoch
    best_epoch = int(np.argmax(history_df["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history_df["val_accuracy"]))

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Accuracy:", test_acc)

    # =====================================================
    # PREDICTIONS
    # =====================================================
    print("\nGenerating confusion matrix...")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(cm, class_names, os.path.join(RESULTS_PATH, f"confusion_matrix_{BACKBONE}.png"))
    np.save(os.path.join(RESULTS_PATH, f"confusion_matrix_{BACKBONE}.npy"), cm)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(RESULTS_PATH, f"confusion_matrix_{BACKBONE}.csv"), index=True)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # =====================================================
    # SAVE METRICS + MODEL
    # =====================================================
    metrics = {
        "backbone": BACKBONE,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "classification_report": report
    }

    with open(os.path.join(RESULTS_PATH, f"metrics_{BACKBONE}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    final_model_path = os.path.join(MODELS_PATH, f"final_model_{BACKBONE}.keras")
    model.save(final_model_path)

    print("\nDONE!")
    print("Saved:")
    print("Best model:", ckpt_path)
    print("Final model:", final_model_path)
    print("Results in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
'''







'''#RESNET
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


# =====================================================
# PATHS
# =====================================================
DATASET_PATH = r"data1/processed/final_3class"

MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
SEED = 42

MODEL_NAME = "ResNet50"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# =====================================================
# UTILS
# =====================================================
def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_plots(history_df, results_path, model_name):
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy")
    plt.title(f"Accuracy Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, f"accuracy_curve_{model_name}.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, f"loss_curve_{model_name}.png"))
    plt.close()


# =====================================================
# MAIN
# =====================================================
def main():
    print("Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    print("Model:", MODEL_NAME)

    # Save class names
    with open(os.path.join(MODELS_PATH, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # =====================================================
    # MODEL (ResNet50)
    # =====================================================
    print("\nBuilding ResNet50 model...")

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ])

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)

    # ResNet preprocess
    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Save model summary
    summary_path = os.path.join(RESULTS_PATH, f"model_summary_{MODEL_NAME}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    # =====================================================
    # CALLBACKS
    # =====================================================
    ckpt_path = os.path.join(MODELS_PATH, f"best_model_{MODEL_NAME}.keras")

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=3, factor=0.2, monitor="val_loss"),
        CSVLogger(os.path.join(RESULTS_PATH, f"training_history_{MODEL_NAME}.csv"))
    ]

    # Save run config
    run_config = {
        "dataset_path": DATASET_PATH,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_stage1": EPOCHS_STAGE1,
        "epochs_stage2": EPOCHS_STAGE2,
        "seed": SEED,
        "base_model": MODEL_NAME,
        "optimizer_stage1": "Adam",
        "lr_stage1": 1e-3,
        "optimizer_stage2": "Adam",
        "lr_stage2": 1e-5
    }

    with open(os.path.join(RESULTS_PATH, f"run_config_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # =====================================================
    # STAGE 1 TRAINING
    # =====================================================
    print("\nStage 1 Training (Frozen ResNet50)...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks
    )

    # =====================================================
    # STAGE 2 TRAINING (Fine-tuning)
    # =====================================================
    print("\nStage 2 Training (Fine-tuning last layers)...")

    base_model.trainable = True

    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks
    )

    # =====================================================
    # COMBINE HISTORY
    # =====================================================
    hist1 = pd.DataFrame(history1.history)
    hist2 = pd.DataFrame(history2.history)

    hist1["stage"] = "stage1"
    hist2["stage"] = "stage2"

    history_df = pd.concat([hist1, hist2], ignore_index=True)
    history_df.to_csv(os.path.join(RESULTS_PATH, f"full_history_{MODEL_NAME}.csv"), index=False)

    save_training_plots(history_df, RESULTS_PATH, MODEL_NAME)

    best_epoch = int(np.argmax(history_df["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history_df["val_accuracy"]))

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Accuracy:", test_acc)

    # =====================================================
    # CONFUSION MATRIX + REPORT
    # =====================================================
    print("\nGenerating confusion matrix...")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    cm_png = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.png")
    cm_npy = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.npy")
    cm_csv = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.csv")

    save_confusion_matrix(cm, class_names, cm_png)
    np.save(cm_npy, cm)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv, index=True)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # =====================================================
    # SAVE METRICS + MODEL
    # =====================================================
    metrics = {
        "model": MODEL_NAME,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "classification_report": report
    }

    with open(os.path.join(RESULTS_PATH, f"metrics_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    final_model_path = os.path.join(MODELS_PATH, f"final_model_{MODEL_NAME}.keras")
    model.save(final_model_path)

    print("\nDONE!")
    print("Best model:", ckpt_path)
    print("Final model:", final_model_path)
    print("Results in:", RESULTS_PATH)


if __name__ == "__main__":
    main()
'''














'''import os
import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

import pandas as pd


# =====================================================
# PATHS
# =====================================================
DATASET_PATH = r"data1/processed/final_4class"
MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
SEED = 42

MODEL_NAME = "ResNet50_4Class"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# =====================================================
# UTILS
# =====================================================
def save_confusion_matrix(cm, class_names, save_path):
    """
    Saves confusion matrix as an image with values.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_plots(history_df, results_path, model_name):
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Train Accuracy")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy")
    plt.title(f"Accuracy Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, f"accuracy_curve_{model_name}.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Train Loss")
    plt.plot(history_df["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_path, f"loss_curve_{model_name}.png"))
    plt.close()


def save_classification_report(report_dict, save_csv_path):
    """
    Converts sklearn classification_report output_dict=True into a clean CSV.
    """
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(save_csv_path, index=True)
    return df


def get_class_weights_from_dataset(train_ds, num_classes):
    """
    Computes class weights from tf.data dataset (train set).
    Helps with imbalance (your fungal is huge usually).
    """
    y_all = []
    for _, y in train_ds:
        y_all.append(y.numpy())
    y_all = np.concatenate(y_all, axis=0)

    counts = np.bincount(y_all, minlength=num_classes)
    total = np.sum(counts)

    class_weights = {}
    for i in range(num_classes):
        if counts[i] == 0:
            class_weights[i] = 1.0
        else:
            class_weights[i] = total / (num_classes * counts[i])

    return class_weights, counts


# =====================================================
# MAIN
# =====================================================
def main():
    print("Loading dataset...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)
    print("Model:", MODEL_NAME)

    # Save class names
    with open(os.path.join(MODELS_PATH, f"class_names_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # =====================================================
    # CLASS WEIGHTS (IMPORTANT)
    # =====================================================
    class_weights, class_counts = get_class_weights_from_dataset(train_ds, len(class_names))

    print("\n📌 Class distribution in TRAIN:")
    for i, name in enumerate(class_names):
        print(f"  {name:10s}: {class_counts[i]}  | weight={class_weights[i]:.3f}")

    # =====================================================
    # MODEL (ResNet50)
    # =====================================================
    print("\nBuilding ResNet50 model...")

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ])

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)

    # ResNet preprocess
    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # Save model summary
    summary_path = os.path.join(RESULTS_PATH, f"model_summary_{MODEL_NAME}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    # =====================================================
    # CALLBACKS
    # =====================================================
    ckpt_path = os.path.join(MODELS_PATH, f"best_model_{MODEL_NAME}.keras")

    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(patience=3, factor=0.2, monitor="val_loss"),
        CSVLogger(os.path.join(RESULTS_PATH, f"training_history_{MODEL_NAME}.csv"))
    ]

    # Save run config
    run_config = {
        "dataset_path": DATASET_PATH,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_stage1": EPOCHS_STAGE1,
        "epochs_stage2": EPOCHS_STAGE2,
        "seed": SEED,
        "base_model": MODEL_NAME,
        "optimizer_stage1": "Adam",
        "lr_stage1": 1e-3,
        "optimizer_stage2": "Adam",
        "lr_stage2": 1e-5,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts_train": {class_names[i]: int(class_counts[i]) for i in range(len(class_names))},
        "class_weights_used": {class_names[i]: float(class_weights[i]) for i in range(len(class_names))}
    }

    with open(os.path.join(RESULTS_PATH, f"run_config_{MODEL_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)

    # =====================================================
    # STAGE 1 TRAINING
    # =====================================================
    print("\nStage 1 Training (Frozen ResNet50)...")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks,
        class_weight=class_weights  # ✅ IMPORTANT
    )

    # =====================================================
    # STAGE 2 TRAINING (Fine-tuning)
    # =====================================================
    print("\nStage 2 Training (Fine-tuning last layers)...")

    base_model.trainable = True

    # Unfreeze last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        callbacks=callbacks,
        class_weight=class_weights  # ✅ IMPORTANT
    )

    # =====================================================
    # COMBINE HISTORY
    # =====================================================
    hist1 = pd.DataFrame(history1.history)
    hist2 = pd.DataFrame(history2.history)

    hist1["stage"] = "stage1"
    hist2["stage"] = "stage2"

    history_df = pd.concat([hist1, hist2], ignore_index=True)
    history_df.to_csv(os.path.join(RESULTS_PATH, f"full_history_{MODEL_NAME}.csv"), index=False)

    save_training_plots(history_df, RESULTS_PATH, MODEL_NAME)

    best_epoch = int(np.argmax(history_df["val_accuracy"]) + 1)
    best_val_acc = float(np.max(history_df["val_accuracy"]))

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Accuracy:", test_acc)

    # =====================================================
    # CONFUSION MATRIX + REPORT
    # =====================================================
    print("\nGenerating confusion matrix + per-class metrics...")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    cm_png = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.png")
    cm_npy = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.npy")
    cm_csv = os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.csv")

    save_confusion_matrix(cm, class_names, cm_png)
    np.save(cm_npy, cm)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv, index=True)

    # =====================================================
    # CLASSIFICATION REPORT (per-class precision/recall/F1)
    # =====================================================
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_csv = os.path.join(RESULTS_PATH, f"classification_report_{MODEL_NAME}.csv")
    report_df = save_classification_report(report_dict, report_csv)

    # Balanced accuracy (important for imbalanced dataset)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Extract per-class metrics cleanly
    per_class_df = report_df.loc[class_names][["precision", "recall", "f1-score", "support"]]
    per_class_df.to_csv(os.path.join(RESULTS_PATH, f"per_class_metrics_{MODEL_NAME}.csv"))

    print("\n📌 Per-class metrics (TEST SET):")
    print(per_class_df)

    print("\n📌 Overall metrics:")
    print("  Test Accuracy         :", float(test_acc))
    print("  Balanced Accuracy     :", float(bal_acc))
    print("  Macro Avg F1-score    :", float(report_dict["macro avg"]["f1-score"]))
    print("  Weighted Avg F1-score :", float(report_dict["weighted avg"]["f1-score"]))

    # =====================================================
    # SAVE METRICS + MODEL
    # =====================================================
    metrics = {
        "model": MODEL_NAME,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "balanced_accuracy": float(bal_acc),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = os.path.join(RESULTS_PATH, f"metrics_{MODEL_NAME}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    final_model_path = os.path.join(MODELS_PATH, f"final_model_{MODEL_NAME}.keras")
    model.save(final_model_path)

    print("\n✅ DONE!")
    print("Best model:", ckpt_path)
    print("Final model:", final_model_path)
    print("Metrics JSON:", metrics_path)
    print("Classification report CSV:", report_csv)
    print("Per-class metrics CSV:", os.path.join(RESULTS_PATH, f"per_class_metrics_{MODEL_NAME}.csv"))
    print("Confusion matrix CSV:", cm_csv)
    print("Results in:", RESULTS_PATH)



if __name__ == "__main__":
    main()
'''













'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# =====================================================
# PATHS
# =====================================================
DATASET_PATH = r"data1/processed/final_4class"
MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
SEED = 42

MODEL_NAME = "ResNet50_4Class_Pathogen"

# =====================================================
# LOAD DATA
# =====================================================
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================================
# CLASS WEIGHTS
# =====================================================
y_all = []
for _, y in train_ds:
    y_all.append(y.numpy())

y_all = np.concatenate(y_all)
counts = np.bincount(y_all, minlength=num_classes)
total = np.sum(counts)

class_weights = {
    i: total / (num_classes * counts[i]) if counts[i] > 0 else 1.0
    for i in range(num_classes)
}

# =====================================================
# MODEL
# =====================================================
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
])

inputs = layers.Input(shape=(224, 224, 3))
x = data_aug(inputs)
x = tf.keras.applications.resnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# =====================================================
# CALLBACKS
# =====================================================
checkpoint_path = os.path.join(MODELS_PATH, f"best_model_{MODEL_NAME}.keras")

callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=3, factor=0.2, monitor="val_loss"),
    CSVLogger(os.path.join(RESULTS_PATH, f"training_log_{MODEL_NAME}.csv"))
]

# =====================================================
# STAGE 1 TRAINING
# =====================================================
print("\nStage 1 Training...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# STAGE 2 TRAINING
# =====================================================
print("\nStage 2 Fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# COMBINE HISTORY
# =====================================================
hist1 = pd.DataFrame(history1.history)
hist2 = pd.DataFrame(history2.history)

hist1["stage"] = "Stage1"
hist2["stage"] = "Stage2"

history_df = pd.concat([hist1, hist2], ignore_index=True)
history_df.to_csv(os.path.join(RESULTS_PATH, f"full_history_{MODEL_NAME}.csv"), index=False)

# =====================================================
# ACCURACY CURVE
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history_df["accuracy"], label="Train Accuracy")
plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve (4-Class Model)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, f"accuracy_curve_{MODEL_NAME}.png"), dpi=300)
plt.close()

# =====================================================
# LOSS CURVE
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history_df["loss"], label="Train Loss")
plt.plot(history_df["val_loss"], label="Validation Loss")
plt.title("Loss Curve (4-Class Model)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_PATH, f"loss_curve_{MODEL_NAME}.png"), dpi=300)
plt.close()

# =====================================================
# TEST EVALUATION
# =====================================================
print("\nEvaluating on test set...")

test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)

y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(RESULTS_PATH, f"confusion_matrix_{MODEL_NAME}.csv"))

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULTS_PATH, f"classification_report_{MODEL_NAME}.csv"))

balanced_acc = balanced_accuracy_score(y_true, y_pred)

print("\nOverall Test Accuracy:", test_acc)
print("Balanced Accuracy:", balanced_acc)

# =====================================================
# SAVE MODEL
# =====================================================
final_model_path = os.path.join(MODELS_PATH, f"final_model_{MODEL_NAME}.keras")
model.save(final_model_path)

print("\nDONE!")
print("Accuracy curve saved.")
print("Loss curve saved.")
'''






import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger
)

from sklearn.metrics import confusion_matrix, classification_report

# =====================================================
# PATHS
# =====================================================
DATASET_PATH = r"data1/processed/final_4class"
MODELS_PATH = r"models/derma_pathogen"
RESULTS_PATH = r"results/derma_pathogen"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# =====================================================
# SETTINGS (FASTER)
# =====================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

MODEL_NAME = "ResNet50_best"

# =====================================================
# LOAD DATA
# =====================================================
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================================
# MODEL (FAST VERSION)
# =====================================================
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze all layers (no fine tuning)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================
checkpoint_path = os.path.join(MODELS_PATH, f"{MODEL_NAME}.keras")

callbacks = [
    ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=3, restore_best_weights=True),
    CSVLogger(os.path.join(RESULTS_PATH, f"training_log_{MODEL_NAME}.csv"))
]

# =====================================================
# TRAIN
# =====================================================
print("\nTraining...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================================
# PLOT ACCURACY
# =====================================================
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(RESULTS_PATH, "accuracy_curve.png"), dpi=300)
plt.close()

# =====================================================
# TEST EVALUATION
# =====================================================
print("\nEvaluating...")

test_loss, test_acc = model.evaluate(test_ds)

print("Test Accuracy:", test_acc)

# =====================================================
# CONFUSION MATRIX
# =====================================================
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)

cm = confusion_matrix(y_true, y_pred)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

cm_df.to_csv(os.path.join(RESULTS_PATH, "confusion_matrix.csv"))

# =====================================================
# CLASSIFICATION REPORT
# =====================================================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(RESULTS_PATH, "classification_report.csv")
)

print("\nDONE!")
print("Model saved to:", checkpoint_path)