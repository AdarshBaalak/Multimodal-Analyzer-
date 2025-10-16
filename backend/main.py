import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import pytesseract
import torch
from torchvision import models, transforms as T
from transformers import pipeline
import requests


# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Multimodal Analyzer 2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ MODELS ------------------
# NLP
sentiment_pipeline = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
zero_shot = pipeline("zero-shot-classification")

# Image Classification
img_model = models.resnet18(weights="IMAGENET1K_V1")
img_model.eval()
img_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ImageNet labels
def load_imagenet_labels() -> List[str]:
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.splitlines()
    except:
        return [f"Unknown {i}" for i in range(1000)]

IMAGENET_LABELS = load_imagenet_labels()

# ------------------ HELPERS ------------------
def pil_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def classify_image(img: Image.Image):
    try:
        img_tensor = img_transform(img).unsqueeze(0)
        with torch.no_grad():
            out = img_model(img_tensor)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            top5 = torch.topk(probs, 5)
            results = []
            for idx, score in zip(top5.indices.tolist(), top5.values.tolist()):
                label = IMAGENET_LABELS[idx] if 0 <= idx < len(IMAGENET_LABELS) else f"Unknown {idx}"
                results.append(f"{label}: {score:.3f}")
            return results
    except:
        return ["Classification failed"]

def perform_ocr(img: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(img).replace("\n", " ").strip()
        return text if text else "No text detected"
    except:
        return "OCR failed"

TOXIC_WORDS = {"hate", "stupid", "idiot", "kill", "bitch", "suck"}
def compute_toxicity_score(text: str) -> float:
    if not text:
        return 0.0
    words = text.lower().split()
    toxic_count = sum(1 for w in words if any(t in w for t in TOXIC_WORDS))
    return round(min(1.0, toxic_count / max(1, len(words)) * 5), 3)

def generate_automated_response(text, sentiment, ocr_text, img_labels):
    if sentiment.lower().startswith("neg"):
        return "We’re sorry about your negative experience. We’ll look into it."
    if "love" in text.lower() and any("box" in lbl.lower() or "carton" in lbl.lower() for lbl in img_labels):
        return "Thanks for the love! We’re glad you liked the product packaging."
    if compute_toxicity_score(text) > 0.5 or compute_toxicity_score(ocr_text) > 0.5:
        return "⚠️ Warning: Toxic content detected in your input."
    return "Thanks for your input! Here’s what we found."

# ------------------ RESPONSE MODEL ------------------
class AnalyzeResult(BaseModel):
    text_sentiment: str
    text_summary: Optional[str]
    topic: Optional[str]
    image_labels: List[str]
    ocr_text: Optional[str]
    toxicity_score: float
    automated_response: str

# ------------------ ENDPOINT ------------------
@app.post("/analyze", response_model=AnalyzeResult)
async def analyze(text: str = Form(...), image: UploadFile = File(None)):
    pil_img = pil_from_upload(image) if image else None

    # Sentiment
    try:
        sent = sentiment_pipeline(text[:1000])[0]
        sentiment_label = sent["label"]
    except:
        sentiment_label = "UNKNOWN"

    # Summarization
    try:
        summary = summarizer(text, max_length=60, min_length=10, do_sample=False)[0]["summary_text"]
    except:
        summary = None

    # Topic classification
    try:
        topic_res = zero_shot(text, ["news", "review", "comment", "complaint", "praise", "question"])
        topic = topic_res["labels"][0]
    except:
        topic = None

    # Image classification
    image_labels = classify_image(pil_img) if pil_img else []

    # OCR
    ocr_text = perform_ocr(pil_img) if pil_img else ""

    # Toxicity
    tox_score = max(compute_toxicity_score(text), compute_toxicity_score(ocr_text))

    # Automated response
    automated = generate_automated_response(text, sentiment_label, ocr_text, image_labels)

    return AnalyzeResult(
        text_sentiment=sentiment_label,
        text_summary=summary,
        topic=topic,
        image_labels=image_labels,
        ocr_text=ocr_text,
        toxicity_score=tox_score,
        automated_response=automated
    )

# ------------------ RUN ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
