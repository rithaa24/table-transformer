# app.py - YOLOv9 Table Detector (Side-by-Side UI)

import io
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv9 Table Detection", layout="wide")
st.title("YOLOv9 Table Detector")

# File uploader
uploaded = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)
st.caption("Upload an image to detect tables. Input (left) → Output (right).")

# Load model once (hardcoded path)
@st.cache_resource(show_spinner=False)
def load_model(_):
    model = YOLO("best.pt")
    return YOLO(model)

# Draw bounding boxes
def draw_boxes(image_pil, boxes_xyxy, labels, color=(255, 0, 0), width=3):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab in zip(boxes_xyxy, labels):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        bbox = draw.textbbox((0, 0), lab, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), lab, fill=(255, 255, 255), font=font)

    return img

# Main inference
if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Loading YOLOv9 model..."):
        model = load_model(None)

    with st.spinner("Running inference..."):
        results = model.predict(image, verbose=False)

    r = results[0]
    names = r.names
    boxes = r.boxes

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Output")
        if boxes is None or len(boxes) == 0:
            st.warning("No detections found.")
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            keep_idx = [i for i, c in enumerate(cls) if "table" in names.get(int(c), "").lower()]

            if len(keep_idx) == 0:
                st.info("No 'table' detections found with current model.")
                annotated_np = r.plot()[:, :, ::-1]  # BGR → RGB
                st.image(annotated_np, use_column_width=True)
            else:
                sel_xyxy = xyxy[keep_idx]
                sel_labels = [f"{names[int(cls[i])]} {confs[i]:.2f}" for i in keep_idx]
                annotated = draw_boxes(image, sel_xyxy, sel_labels, color=(255, 0, 0), width=3)
                st.image(annotated, use_column_width=True)

                # Download button
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                st.download_button(
                    "Download Annotated Image",
                    data=buf.getvalue(),
                    file_name="yolov9_tables.png",
                    mime="image/png",
                )
