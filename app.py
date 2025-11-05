import gradio as gr
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont

# ----------------------------
# Model loading
# ----------------------------
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------------
# Utilities
# ----------------------------
def draw_boxes(pil_img, boxes, labels, scores):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw = ImageDraw.Draw(pil_img)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 122, 255), width=4)
        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 8, y_text + th + 4], fill=(0, 122, 255))
        draw.text((x1 + 4, y_text + 2), text, fill=(255, 255, 255), font=font)
    return pil_img


def detect(image, threshold):
    if image is None:
        return None, []

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    boxes = results["boxes"].tolist()
    labels_ids = results["labels"].tolist()
    scores = results["scores"].tolist()
    labels = [model.config.id2label[i] for i in labels_ids]

    vis = image.copy()
    vis = draw_boxes(vis, boxes, labels, scores)

    rows = []
    for label, score, (x1, y1, x2, y2) in zip(labels, scores, boxes):
        rows.append([
            label,
            round(float(score), 4),
            round(float(x1), 2),
            round(float(y1), 2),
            round(float(x2), 2),
            round(float(y2), 2),
        ])

    return vis, rows


# ----------------------------
# Load external CSS
# ----------------------------
with open("static/style.css") as f:
    custom_css = f.read()

# ----------------------------
# UI Layout
# ----------------------------
demo = gr.Blocks(css=custom_css, title="AI Objects Guesser")

with demo:
    with gr.Row(elem_classes=["section-hero"]):
        gr.HTML("""
        <div>
          <span class="badge">🧠 AI Objects Guesser</span>
          <h1>Upload an image and set a confidence threshold</h1>
          <p>Detect objects in your images with state-of-the-art AI technology.</p>
        </div>
        """)

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["card"]):
                gr.HTML("<h3>Image</h3>")
                inp = gr.Image(
                    type="pil",
                    label="",
                    height=360,
                    elem_id="input-image",
                    value="https://www.setupgaming.fr/wp-content/uploads/elementor/thumbs/Comment-creer-un-setup-gaming-sans-depenser-une-fortune-Astuces-pour-un-setup-gratuit-SetupGaming-scaled-r0m2hndozsa8gy65m8fzeijrn5wydq4z3tzx4b7ib4.jpg"
                )
                with gr.Row():
                    thr = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="Confidence Threshold")
                detect_btn = gr.Button("🔍 Detect Objects", elem_id="detect-btn", elem_classes=["btn-primary"])

        with gr.Column(scale=6):
            with gr.Group(elem_classes=["card"]):
                gr.HTML("<h3>Detections</h3>")
                out_img = gr.Image(type="pil", label="", height=360)
                out_tbl = gr.Dataframe(
                    headers=["Label", "Score", "X1", "Y1", "X2", "Y2"],
                    label="Detections (table)",
                    interactive=False,
                    wrap=True,
                )

    detect_btn.click(detect, inputs=[inp, thr], outputs=[out_img, out_tbl])


if __name__ == "__main__":
    # Optionnel: réduire le bruit des logs Transformers
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
