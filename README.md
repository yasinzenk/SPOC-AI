# 🧠 AI Objects Guesser (Python)

This project is an **AI-powered object detection service** built with **Python** using **Gradio**, **Transformers**, and **PyTorch**.  
It allows users to upload an image and automatically detect objects using the **DETR (DEtection TRansformer)** model by **Facebook / Meta AI**.

---

## 🚀 Features

- 🖼 Upload or paste any image (JPG, PNG, etc.)  
- 🎯 Detect multiple objects with bounding boxes  
- 🔍 Adjustable confidence threshold (slider UI)  
- 📊 Display of detection results in a clean table (label, score, coordinates)  
- 🎨 Custom CSS design for a modern interface  
- ⚙️ Runs locally via `Gradio`, or deployable to Hugging Face Spaces  

---

## 🧠 Detection Logic

For each uploaded image:

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
| 💻 **Gradio** | Web-based UI for model interaction |
| ⚡ **FastAPI + Uvicorn** | Backend infrastructure (through Gradio) |
| 🎨 **Custom CSS** | Enhances the frontend look & feel |

---

## 📦 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ton-utilisateur/spoc_ai
cd spoc_ai
