# 🧠 AI Objects Guesser (Python)

This project is an **AI-powered object detection service** built with **Python** using **Gradio**, **Transformers**, and **PyTorch**.  
It allows users to upload an image **or use their webcam live** to automatically detect objects using the **DETR (DEtection TRansformer)** model by **Facebook / Meta AI**.

---

## 🚀 Features

- 🖼 Upload an image or use your **webcam** in real time  
- 🎯 Detect multiple objects with bounding boxes  
- 🔍 Adjustable confidence threshold (slider UI)  
- 📊 Display of detection results in a clean table (label, score, coordinates)  
- ⚙️ Runs locally via `Gradio`, or deployable to Hugging Face Spaces  

---

## 🧠 Detection Logic

For each uploaded or captured image:

1. The image is processed using **DETR’s pre-trained ResNet-50 backbone**.  
2. **Transformers** generate attention-based predictions for each object.  
3. Results are filtered using a confidence threshold.  
4. **Bounding boxes** and **labels** are drawn directly on the image.  
5. A **summary table** lists all detected objects with coordinates and confidence scores.

---

## 🧰 Stack

| Tool / Library | Role |
|----------------|------|
| 🧠 **DETR (facebook/detr-resnet-50)** | Transformer-based object detection model |
| 🤗 **Transformers** | Loads and runs pre-trained models from Hugging Face |
| 🔥 **PyTorch** | Deep learning framework for inference |
| 💻 **Gradio** | Web-based UI for model interaction (image + webcam) |
| ⚡ **FastAPI + Uvicorn** | Backend infrastructure (through Gradio) |

---

## 📦 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ton-utilisateur/spoc_ai
cd spoc_ai
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Your `requirements.txt` should contain at least:

```txt
fastapi
uvicorn
httpx
pydantic
torch
transformers
gradio
pillow

Then install everything with:
```bash
pip install -r requirements.txt
```

### ▶️ Run the application locally

Once installed, simply run:

```bash
python app.py

You’ll see output like this:

Running on local URL: http://0.0.0.0:7860
Running on public URL: https://xxxxxx.gradio.live
```

## 🖼 How It Works (Step by Step)

1. Upload an image or use your **webcam**.  
2. Choose a **confidence threshold** (default = 0.9).  
3. Click **"🔍 Detect Objects"**.  
4. DETR analyzes the image and returns:
   - An annotated image with bounding boxes.  
   - A table listing all detected objects with their scores and coordinates.

## 🧩 Project Structure

spoc_ai/
│
├── app.py # Main Gradio application
├── requirements.txt # Python dependencies
├── static/
│ └── style.css # Custom frontend styles
└── README.md # Project documentation

## 💡 Next Improvements

- 🚀 Add support for **image batch processing**  
- 💾 Save results (image + detections) locally or in a database  
- 🌐 Deploy permanently on **Hugging Face Spaces**  
- 🧩 Add **multi-model selection** (YOLO, DETR, SAM)  
- 📱 Make the UI mobile-friendly  
