import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import time
import imutils

# ----------------------------- Core ML functions -----------------------------

def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return image
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]


def process_scan(image, image_size):
    X = []
    w, h = image_size
    image = crop_brain_contour(image)
    image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    X.append(image)
    return np.array(X)


def analyze_brain_scan(img_array, model_type="cnn"):
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    X = process_scan(img_array, (IMG_WIDTH, IMG_HEIGHT))

    if model_type == "cnn":
        model_path = "./Brain_Tumor_Classification_Models/final_cnn_model.keras"
        if not os.path.exists(model_path):
            model_path = "./Brain_Tumor_Detection_Models/cnn-parameters-improvement-01-0.88.keras"
    else:
        model_path = "./Brain_Tumor_Classification_Models/final_hybrid_model.keras"

    model = load_model(filepath=model_path)

    start_time = time.time()
    prediction = model.predict(X, verbose=0)
    end_time = time.time()

    class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx] * 100

    return {
        "prediction": class_names[predicted_class_idx],
        "confidence": confidence,
        "all_confidences": {class_names[i]: prediction[0][i] * 100 for i in range(len(class_names))},
        "processing_time": end_time - start_time,
        "model_type": "CNN" if model_type == "cnn" else "Hybrid",
    }


TUMOR_INFO = {
    "Glioma": "Arises from glial cells; can be low or high grade. High-grade forms like glioblastoma are aggressive. Treatment: surgery + radiation/chemo.",
    "Meningioma": "Develops from the meninges covering the brain/spinal cord. Usually benign, slow-growing. Treatment: observation, surgery, or radiation.",
    "Pituitary": "Forms in the pituitary gland; may affect hormone production. Usually benign. Treatment: medication, surgery, or radiation.",
    "No Tumor": "No tumorous tissue detected in the scan. Always confirm with a qualified radiologist.",
}

TUMOR_ICON = {"Glioma": "🔴", "Meningioma": "🟠", "Pituitary": "🟣", "No Tumor": "🟢"}

# ----------------------------- Page config -----------------------------

st.set_page_config(
    page_title="Brain Tumor Classification System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠",
)

# ----------------------------- CSS -----------------------------

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1500px;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', Inter, sans-serif; }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg,#4f8bff,#7c5cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #9aa4b2;
        margin-bottom: 1.1rem;
        line-height: 1.4;
    }

    .panel {
        background: #161b26;
        border: 1px solid #262c3a;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        height: 100%;
    }
    .panel-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #dfe4ec;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .result-card {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid;
    }
    .result-card.positive { background: rgba(255,70,70,0.08); border-color: rgba(255,70,70,0.4); }
    .result-card.negative { background: rgba(50,220,120,0.08); border-color: rgba(50,220,120,0.4); }

    .result-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: #9aa4b2; margin-bottom: 0.2rem; }
    .result-value { font-size: 1.7rem; font-weight: 800; }
    .result-value.positive { color: #ff5c5c; }
    .result-value.negative { color: #35d488; }

    .conf-num { font-size: 2.1rem; font-weight: 800; color: #f2f4f8; line-height: 1; }
    .conf-caption { font-size: 0.78rem; color: #9aa4b2; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem;}

    .bar-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.35rem 0; }
    .bar-label { width: 92px; font-size: 0.82rem; color: #cfd5e0; flex-shrink: 0; }
    .bar-track { flex: 1; background: #262c3a; border-radius: 6px; height: 14px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 6px; }
    .bar-pct { width: 46px; text-align: right; font-size: 0.8rem; color: #cfd5e0; flex-shrink: 0; }

    .meta-row { display:flex; gap: 1.2rem; font-size: 0.8rem; color: #9aa4b2; margin-top: 0.6rem; border-top: 1px solid #262c3a; padding-top: 0.6rem;}
    .meta-row b { color: #dfe4ec; }

    .info-box {
        background: #12161f;
        border-left: 3px solid #4f8bff;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        font-size: 0.82rem;
        color: #c3cad6;
        margin-top: 0.7rem;
        line-height: 1.45;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg,#4f8bff,#7c5cff);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 0;
        font-size: 0.95rem;
        transition: 0.15s ease;
    }
    .stButton>button:hover { filter: brightness(1.12); transform: translateY(-1px); }

    section[data-testid="stFileUploader"] { padding: 0; }
    section[data-testid="stFileUploader"] > div { border-radius: 12px; }

    .stImage img { border-radius: 12px; border: 1px solid #262c3a; }

    [data-testid="stSidebar"] { background: #10141c; }
    .placeholder-box {
        display:flex; align-items:center; justify-content:center;
        height: 340px; border: 2px dashed #262c3a; border-radius: 14px;
        color: #6b7383; font-size: 0.95rem; text-align:center; padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Sidebar -----------------------------

st.sidebar.markdown("## 🧠 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Classification Model",
    ["CNN Model", "Hybrid Model"],
    help="CNN Model is faster but may be less accurate. Hybrid Model combines multiple architectures for better performance.",
)
selected_model = "cnn" if model_type == "CNN Model" else "hybrid"

with st.sidebar.expander("ℹ️ About the Models"):
    st.markdown("""
    **CNN Model** — a convolutional network trained specifically for brain tumor classification.

    **Hybrid Model** — combines pretrained VGG16 + ResNet50 features with a custom CNN for improved accuracy.
    """)

with st.sidebar.expander("📖 About Brain Tumors"):
    st.markdown("""
    **Glioma** — from glial cells; can be low or high grade.

    **Meningioma** — from the meninges; mostly benign.

    **Pituitary** — in the pituitary gland; affects hormones.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("⚕️ For research/educational use only. Not a substitute for professional medical diagnosis.")

# ----------------------------- Header -----------------------------

st.markdown('<div class="app-title">🧠 ADVANCED BRAIN TUMOR CLASSIFICATION SYSTEM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">AI-powered MRI analysis to detect and classify Glioma, Meningioma, '
    'Pituitary tumors, or confirm a tumor-free scan — instantly.</div>',
    unsafe_allow_html=True,
)

# ----------------------------- Main layout -----------------------------

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PNG, JPEG, JPG — max 200MB",
        type=["png", "jpeg", "jpg"],
        label_visibility="collapsed",
    )

    opencv_image = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR", use_container_width=True, caption=uploaded_file.name)
        analyze_clicked = st.button("🔍  Analyze Scan", key="analyze_button")
    else:
        st.markdown('<div class="placeholder-box">Drag & drop or browse to upload an MRI scan<br>to begin analysis</div>', unsafe_allow_html=True)
        analyze_clicked = False
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📊 Analysis Result</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown('<div class="placeholder-box">Results will appear here after you<br>upload and analyze a scan</div>', unsafe_allow_html=True)

    elif not analyze_clicked and "last_result" not in st.session_state:
        st.info("Scan loaded. Click **Analyze Scan** to run classification.")

    else:
        if analyze_clicked:
            with st.spinner("Running model inference..."):
                results = analyze_brain_scan(opencv_image, selected_model)
                st.session_state["last_result"] = results
        results = st.session_state.get("last_result")

        if results:
            is_negative = results["prediction"] == "No Tumor"
            card_cls = "negative" if is_negative else "positive"
            icon = "✅" if is_negative else "⚠️"

            colA, colB = st.columns([1.3, 1])
            with colA:
                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div class="result-label">Prediction Result</div>
                    <div class="result-value {card_cls}">{icon} {results['prediction']}</div>
                </div>
                """, unsafe_allow_html=True)
            with colB:
                st.markdown(f"""
                <div class="result-card {card_cls}">
                    <div class="conf-caption">Confidence</div>
                    <div class="conf-num">{results['confidence']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='font-size:0.85rem; color:#9aa4b2; margin: 0.3rem 0 0.4rem;'>CLASS PROBABILITIES</div>", unsafe_allow_html=True)

            colors = {"Glioma": "#ff5c5c", "Meningioma": "#ffb648", "Pituitary": "#a06bff", "No Tumor": "#35d488"}
            bars_html = ""
            for cls, val in sorted(results["all_confidences"].items(), key=lambda x: -x[1]):
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-label">{cls}</div>
                    <div class="bar-track"><div class="bar-fill" style="width:{val}%; background:{colors[cls]};"></div></div>
                    <div class="bar-pct">{val:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="meta-row">
                <div>Model: <b>{results['model_type']}</b></div>
                <div>Inference time: <b>{results['processing_time']:.3f}s</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
                {TUMOR_ICON[results['prediction']]} <b>{results['prediction']}:</b> {TUMOR_INFO[results['prediction']]}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
