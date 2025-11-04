# api/index.py
from fastapi import FastAPI
import gradio as gr
from app import demo  # importe l'objet 'demo'

app = FastAPI()

# Monte Gradio à la racine
app = gr.mount_gradio_app(app, demo, path="/")

# Petit healthcheck pour debug
@app.get("/api/health")
def health():
    return {"ok": True}
