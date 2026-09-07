import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
from datetime import datetime

# ============================================================
# SMART CASSAVA DISEASE DETECTION SYSTEM (Web Version)
# ============================================================

MODEL_PATH = "cassava_mobilenetv2_with_cam_simplified_float32.tflite"
LABELS_PATH = "labels.txt"
WEIGHTS_PATH = "classifier_weights.json"

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISEASE_INFO = {
    "Cassava Bacterial Blight (CBB)":
        "A bacterial disease causing angular leaf spots, wilting, and gum exudation on stems. "
        "Management: use disease-free cuttings, remove and burn infected plants, avoid working "
        "in fields when wet.",
    "Cassava Brown Streak Disease (CBSD)":
        "A viral disease causing brown streaks on stems and yellow patches on leaves, often "
        "damaging the storage roots. Management: plant certified virus-free cuttings, control "
        "whitefly populations, remove infected plants early.",
    "Cassava Green Mite (CGM)":
        "A pest causing yellow mottling, leaf curling, and stunted growth from mite feeding. "
        "Management: introduce natural predator mites, plant resistant varieties, avoid drought "
        "stress on plants.",
    "Cassava Mosaic Disease (CMD)":
        "A viral disease spread by whiteflies, causing mottled yellow-green patterns and leaf "
        "distortion. Management: use disease-free planting material, remove infected plants "
        "promptly, control whitefly vectors.",
    "Healthy":
        "No signs of disease detected. Continue routine monitoring and good field hygiene to "
        "maintain plant health.",
}

# ============================================================
# LOAD MODEL (cached so it only loads once per session)
# ============================================================

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f if line.strip()]

    with open(WEIGHTS_PATH) as f:
        weights_data = json.load(f)
    classifier_weights = np.array(weights_data["weight"])  # [5, 1280]

    return interpreter, labels, classifier_weights


interpreter, class_names, classifier_weights = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Figure out which output is logits [1,5] vs feature_maps [1,7,7,1280]
logits_index = None
features_index = None
for detail in output_details:
    if detail["shape"][-1] == len(class_names):
        logits_index = detail["index"]
    else:
        features_index = detail["index"]

# ============================================================
# PREPROCESSING + PREDICTION
# ============================================================

def preprocess(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict(image: Image.Image):
    input_data = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    logits = interpreter.get_tensor(logits_index)[0]
    feature_maps = interpreter.get_tensor(features_index)[0]  # [7, 7, 1280]

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    pred_idx = int(np.argmax(probs))

    # ---- Class Activation Map (CAM) ----
    class_weights = classifier_weights[pred_idx]  # [1280]
    cam = np.tensordot(feature_maps, class_weights, axes=([2], [0]))  # [7,7]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return pred_idx, probs, cam


def make_heatmap_overlay(original_image: Image.Image, cam: np.ndarray):
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(
        original_image.size, Image.BILINEAR
    )
    cam_arr = np.array(cam_img).astype(np.float32) / 255.0

    # Colorize: blue (low) -> yellow -> red (high)
    heat_r = cam_arr * 255
    heat_g = np.clip(1 - np.abs(cam_arr - 0.5) * 2, 0, 1) * 255
    heat_b = (1 - cam_arr) * 255
    heatmap_rgb = np.stack([heat_r, heat_g, heat_b], axis=-1)

    orig_arr = np.array(original_image.convert("RGB")).astype(np.float32)
    alpha = (cam_arr * 0.55)[..., None]
    blended = orig_arr * (1 - alpha) + heatmap_rgb * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Cassava Disease Detection",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Smart Cassava Disease Detection")
st.write("Upload a cassava leaf image or take a picture using your camera.")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader(
    "Upload a cassava leaf image", type=["jpg", "jpeg", "png"]
)
camera_image = st.camera_input("Or take a picture of the cassava leaf")

image_file = camera_image if camera_image is not None else uploaded_file

if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    pred_idx, probs, cam = predict(image)
    overlay = make_heatmap_overlay(image, cam)

    st.image(
        overlay,
        caption="Highlighted areas show what most influenced the prediction "
                "(red = strongest influence)",
        use_container_width=True,
    )

    predicted_name = class_names[pred_idx]
    confidence = probs[pred_idx] * 100
    detection_time = datetime.now()

    st.success(f"Prediction: {predicted_name}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.write(f"🕐 Detection Time: {detection_time.strftime('%d/%m/%Y %I:%M:%S %p')}")
    st.write(DISEASE_INFO.get(predicted_name, ""))

    status = "Healthy" if pred_idx == 4 else "Infected"
    if status == "Healthy":
        st.success("🟢 Status: HEALTHY")
    else:
        st.error("🔴 Status: INFECTED")

    st.session_state.history.append({
        "Date & Time": detection_time.strftime("%d/%m/%Y %I:%M:%S %p"),
        "Disease": predicted_name,
        "Confidence": f"{confidence:.2f}%",
        "Status": status,
    })

    st.subheader("Prediction Probabilities")
    for i, cname in enumerate(class_names):
        p = probs[i] * 100
        st.write(f"{cname}: {p:.2f}%")
        st.progress(float(probs[i]))

# ============================================================
# MONITORING DASHBOARD
# ============================================================

st.subheader("📊 Monitoring Dashboard")

total_scans = len(st.session_state.history)
healthy_count = sum(1 for h in st.session_state.history if h["Status"] == "Healthy")
infected_count = sum(1 for h in st.session_state.history if h["Status"] == "Infected")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Scans", total_scans)
with col2:
    st.metric("Healthy", healthy_count)
with col3:
    st.metric("Infected", infected_count)

# ============================================================
# DETECTION HISTORY
# ============================================================

st.subheader("📋 Detection History")

if len(st.session_state.history) > 0:
    st.dataframe(st.session_state.history, use_container_width=True)
else:
    st.info("No detections recorded yet.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("Smart Cassava Disease Detection System")
