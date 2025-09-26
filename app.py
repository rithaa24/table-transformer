# app.py - YOLOv9 Table Detector with Detections Table

import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv9 Table Detection", layout="centered")
st.title("YOLOv9 Table Detector")

# Model path input
model_path = st.text_input(
    "Model weights (path or name)",
    value="C:\\Users\\shari\\Desktop\\Zoho\\Table transformers\\table bank\\best.pt",
    help="Use a YOLOv9 weight file including a 'table' class; named weights auto-download in Ultralytics.",
)

conf = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
imgsz = st.select_slider("Inference size (px)", options=[640, 768, 960, 1280], value=640)
show_all = st.checkbox("Show all detections (not only 'table')", value=False)

uploaded = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)
st.caption("Max upload size is configurable via Streamlit settings if needed.")

# Load model once
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

# Draw bounding boxes on PIL image
def draw_boxes(image_pil, boxes_xyxy, labels, color=(255, 0, 0), width=3):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{lab}"

        # Pillow ≥10 compatible text size
        if font is not None:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        else:
            tw, th = draw.textlength(text), 10  # fallback

        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    return img

# Run inference when image uploaded and model path set
if uploaded and model_path:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input", use_column_width=True)

    with st.spinner("Loading YOLOv9..."):
        model = load_model(model_path)

    with st.spinner("Running inference..."):
        results = model.predict(
            image,
            conf=conf,
            imgsz=int(imgsz),
            verbose=False,
        )

    r = results[0]
    names = r.names  # dict: class_id -> name
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        st.warning("No detections found.")
    else:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # Build a DataFrame for all detections
        data = []
        for i in range(len(cls)):
            x1, y1, x2, y2 = xyxy[i]
            data.append({
                "Class": names[int(cls[i])],
                "Confidence": round(float(confs[i]), 2),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })
        df = pd.DataFrame(data)

        # Show table of detections
        st.subheader("All Detections")
        st.dataframe(df)

        # Filter 'table' class if needed
        if show_all:
            keep_idx = list(range(len(cls)))
        else:
            keep_idx = [i for i, c in enumerate(cls) if "table" in names.get(int(c), "").lower()]

        if len(keep_idx) == 0:
            st.info("No 'table' detections found with current model/classes and threshold.")
            annotated_np = r.plot()[:, :, ::-1]  # BGR → RGB
            st.image(annotated_np, caption="All detections (for reference)", use_column_width=True)
        else:
            sel_xyxy = xyxy[keep_idx]
            sel_labels = [f"{names[int(cls[i])]} {confs[i]:.2f}" for i in keep_idx]
            annotated = draw_boxes(image, sel_xyxy, sel_labels, color=(255, 0, 0), width=3)
            st.image(annotated, caption="Table detections", use_column_width=True)

            # Download annotated image
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                "Download annotated image",
                data=buf.getvalue(),
                file_name="yolov9_tables.png",
                mime="image/png",
            )
