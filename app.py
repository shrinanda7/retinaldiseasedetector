# app.py
import io, os, time, json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
import numpy as np
import google.generativeai as genai

# ---------- Config ----------
HF_REPO_ID = "shrinanda/rfmid-10cls-resnet34"
MODEL_FILE = "best_resnet34.pth"
LABEL_FILE = "label_map.json"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENAI_KEY = os.getenv("GEMINI_API_KEY", "")
# -----------------------------

# download weights + labels (once)
os.makedirs("models", exist_ok=True)
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE, local_dir="models")
label_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LABEL_FILE, local_dir="models")
labels = json.load(open(label_path))

# transforms
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# build & load model
model = models.resnet34(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(labels)))
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE).eval()
print("✅ Model loaded on", DEVICE)

# init Gemini (optional)
if GENAI_KEY:
    genai.configure(api_key=GENAI_KEY)

def predict_img(pil_img):
    t0 = time.time()
    x = eval_tfms(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(out))
    return {
        "pred_label": labels[idx],
        "pred_prob": float(out[idx]),
        "all_probs": {labels[i]: float(p) for i,p in enumerate(out)},
        "latency_ms": int((time.time()-t0)*1000)
    }

def gemini_report(pred_label, prob):
    if not GENAI_KEY: return "Gemini key not set"
    prompt = f'''
    Generate a detailed clinical report for an eye disease screening.

The AI model provided the following output:
Predicted Condition: {pred_label}
Confidence Score: {prob:.2f}

The report must be structured with the following sections:
1.  *AI Finding:* (State the predicted condition and confidence score).
2.  *Condition Overview:* (Provide a brief, 2-3 sentence clinical description of the predicted condition).
3.  *Associated Fundoscopic Features:* (List the key observable biomarkers on a fundus image for this condition).
4.  *Clinical Implications & Risk:* (Describe the patient risks, common symptoms, and risk of progression).
5.  *Recommended Next Steps:* (Outline the standard management, referral, and follow-up protocol).
6.  *Disclaimer:* (Include a standard disclaimer that this is an AI tool and not a final diagnosis).

Format the output clearly for a medical professional.
    
    
    
    
    '''
    g = genai.GenerativeModel("gemini-2.5-pro")
    return g.generate_content(prompt).text.strip()

app = FastAPI(title="RFMiD Inference API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return predict_img(pil)

@app.post("/report")
async def report(file: UploadFile = File(...)):
    pil = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pred = predict_img(pil)
    pred["report"] = gemini_report(pred["pred_label"], pred["pred_prob"])
    return pred
