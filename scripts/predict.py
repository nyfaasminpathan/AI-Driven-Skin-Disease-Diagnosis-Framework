import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

MODEL_PATH = "best_model_ResNet50.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = ["Bacterial", "Fungal", "Viral"]

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()


image_path = "test.jpg"  


if not os.path.exists(image_path):
    print("❌ Image not found. Please check the path.")
else:
    try:
        # Load and resize image
        img = image.load_img(image_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)

        # Expand dimensions (1,224,224,3)
        img_array = np.expand_dims(img_array, axis=0)

        # ResNet50 preprocessing
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)

        print("Raw Predictions:", predictions)

        # Get highest probability index
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[predicted_index]

        print("\n==============================")
        print("      PREDICTION RESULT")
        print("==============================")
        print(f"Category   : {predicted_class}")
        print(f"Confidence : {confidence * 100:.2f}%")
        print("==============================")

    except Exception as e:
        print("❌ Error during prediction:", e)