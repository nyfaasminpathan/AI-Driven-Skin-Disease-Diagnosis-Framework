import streamlit as st
import time
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Analysis Results", layout="wide")

# -----------------------
# SESSION STATE
# -----------------------
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# -----------------------
# LOAD MODELS
# -----------------------
@st.cache_resource
def load_models():
    stage1 = tf.keras.models.load_model("model/best_model.h5")
    stage2 = tf.keras.models.load_model("model/final_model_ResNet50_4Class_Pathogen.keras")
    return stage1, stage2

stage1_model, stage2_model = load_models()

# Stage-2 classes (UPDATED → 4 classes)
stage2_classes = ["Bacterial", "Fungal", "Viral", "Other"]

# -----------------------
# IMAGE PREPROCESS
# -----------------------
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------
# STYLING
# -----------------------
st.markdown("""
<style>

.block-container{
padding-top:2rem;
}

[data-testid="stHeader"]{
display:none;
}

[data-testid="stToolbar"]{
display:none;
}

footer{
display:none;
}

[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#f0fdfa 0%,#f8fafc 100%);
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# CHECK IMAGE
# -----------------------
if 'uploaded_image' not in st.session_state:
    st.warning("No image detected. Please upload an image on the Home page.")
    if st.button("Back to Home"):
        st.switch_page("app.py")
    st.stop()

image = Image.open(st.session_state['uploaded_image'])

# -----------------------
# HEADER
# -----------------------
st.markdown("<h1 style='color:#0f766e'>Analysis Report</h1>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1.2,1], gap="large")

# =================================================
# LEFT SIDE → IMAGE + EDIT
# =================================================
with col1:

    with st.expander("Brightness & Rotation"):

        c1, c2 = st.columns(2)

        with c1:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)

        with c2:
            rotate_angle = st.slider("Rotate", -180, 180, 0, 90)

    enhancer = ImageEnhance.Brightness(image)
    bright_img = enhancer.enhance(brightness)

    adjusted_img = bright_img.rotate(rotate_angle, expand=True)

    adjusted_img.thumbnail((250,250))

    st.markdown("**Drag the bounding box to crop the target area:**")

    crop_col,_ = st.columns([1.5,2])

    with crop_col:

        cropped_img = st_cropper(
            adjusted_img,
            realtime_update=True,
            box_color='#0ea5a4',
            aspect_ratio=None
        )

    with st.expander("View Final Cropped Image", expanded=True):

        st.image(cropped_img, width=150)

    # -----------------------
    # ANALYZE BUTTON
    # -----------------------
    if st.button("Analyze Image"):

        st.session_state.run_analysis = True

    st.write("")

    if st.button("← Scan New Image", use_container_width=True):

        del st.session_state['uploaded_image']
        st.session_state.run_analysis = False
        st.switch_page("app.py")

# =================================================
# RIGHT SIDE → RESULTS
# =================================================
with col2:

    with st.container(border=True):

        st.markdown("### 🤖 AI Classification Results")

        def create_progress_bar(label,percentage,color):

            return f"""
            <div style="background:white;padding:12px;border-radius:12px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;">
            <strong>{label}</strong>
            <span style="color:{color};font-weight:700;">{percentage:.1f}%</span>
            </div>
            <div style="width:100%;background:#f1f5f9;border-radius:8px;height:10px;">
            <div style="width:{percentage}%;background:{color};height:100%;border-radius:8px;"></div>
            </div>
            </div>
            """

        # -----------------------
        # RUN MODEL AFTER BUTTON
        # -----------------------
        if st.session_state.run_analysis:

            img = preprocess(cropped_img)

            with st.status("Running Two-Stage AI Analysis...", expanded=True) as status:

                st.write("Stage 1: Differentiating Healthy vs Diseased tissue...")

                pred1 = stage1_model.predict(img)

                disease_prob = float(pred1[0][0]) * 100
                healthy_prob = 100 - disease_prob

                stage1_probs = [healthy_prob, disease_prob]

                stage1_label = "Diseased" if disease_prob > 50 else "Healthy"

                time.sleep(1.2)

                if stage1_label == "Diseased":

                    st.write("Stage 2: Identifying specific pathogen signatures...")

                    pred2 = stage2_model.predict(img)

                    stage2_probs = pred2[0] * 100

                    predicted_class = stage2_classes[np.argmax(pred2)]

                    time.sleep(1.2)

                status.update(label="Analysis Successful", state="complete")

            # -----------------------
            # STAGE 1 RESULTS
            # -----------------------
            st.markdown("#### Stage 1: Health Assessment")

            st.markdown(create_progress_bar("Diseased", stage1_probs[1], "#f97316"), unsafe_allow_html=True)
            st.markdown(create_progress_bar("Healthy", stage1_probs[0], "#22c55e"), unsafe_allow_html=True)

            st.markdown("---")

            # -----------------------
            # STAGE 2 RESULTS
            # -----------------------
            if stage1_label == "Diseased":

                st.markdown("#### Stage 2: Pathogen Classification")

                if predicted_class == "Other":
                    st.success("Primary Diagnosis: **Other Skin Condition Detected**")
                else:
                    st.success(f"Primary Diagnosis: **{predicted_class} Infection Detected**")

                st.markdown(create_progress_bar("Bacterial", stage2_probs[0], "#0ea5a4"), unsafe_allow_html=True)
                st.markdown(create_progress_bar("Fungal", stage2_probs[1], "#f59e0b"), unsafe_allow_html=True)
                st.markdown(create_progress_bar("Viral", stage2_probs[2], "#ef4444"), unsafe_allow_html=True)
                st.markdown(create_progress_bar("Other", stage2_probs[3], "#6366f1"), unsafe_allow_html=True)

            else:

                st.success("Skin appears **Healthy**. No disease detected.")

            st.warning("⚠️ This is an AI-generated assessment, not a clinical diagnosis.")