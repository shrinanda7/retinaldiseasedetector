Retina Disease Classification and Explainable AI System

Deep learning–based multi-class retinal disease diagnosis using fundus images
Built using ResNet-34, FastAPI, Hugging Face, and Google Gemini API

🌍 Project Overview

This project detects 10 retinal diseases from fundus images using a fine-tuned ResNet-34 model trained on the merged RFMiD + RFMiD 2.0 dataset.
It integrates Explainable AI (Grad-CAM, SHAP, LIME) (upcoming) 
and Gemini for automated report generation.

The backend is designed as a FastAPI microservice, which can be accessed via both web and mobile apps for real-time inference.

🧠 Key Features

✅ Multi-class classification (10 diseases)
✅ Custom-merged RFMiD dataset (single-label version)
✅ Trained ResNet-34 model with 81% validation accuracy, F1 = 0.69
✅ FastAPI REST API for image inference
✅ Integrated Gemini-powered detailed disease report generation
✅ Future integration: XAI visualization (Grad-CAM, SHAP, LIME)
✅ Cloud-deployable (Hugging Face, Oracle, or Render)

🧾 Dataset

Dataset Source: RFMiD
 + RFMiD 2.0
Classes Used (10):

DR, ODC, DN, BRVO, ARMD, CRVO, RPEC, AION, ERM, MHL


You created a strict single-label merged dataset with these splits:

Split	Samples	%
Train	768	70%
Val	165	15%
Test	165	15%

Total images = 1098 fundus images.

🧩 Model Training Summary
Parameter	Value
Base Model	ResNet-34 (pretrained on ImageNet)
Image Size	224×224
Optimizer	Adam (lr=1e-4)
Batch Size	32
Epochs	30
Loss	CrossEntropy
Augmentations	Horizontal Flip, Vertical Flip, Color Jitter
Validation Acc	0.81
Macro F1	0.69

Checkpoint: best_resnet34.pth

☁️ Deployment Architecture
Frontend (Web/Mobile)
       │
       ▼
FastAPI Backend (Docker)
 ├── Loads model from HF once
 ├── /predict → classification
 ├── /report  → Gemini report
       │
       ▼
Gemini API → generates medical-style report


Model hosting: Hugging Face Hub
Backend hosting: locally as of now(Dockerized FastAPI)
Frontend: Flutter / React (image upload + report display)

⚙️ API Endpoints
Endpoint	Method	Description
/healthz	GET	Check service health
/predict	POST	Upload image → returns prediction JSON
/report	POST	Upload image → returns prediction + Gemini report
Example Request
curl -X POST -F "file=@/path/to/fundus.jpg" \
http://<YOUR_SERVER_IP>:8000/report

Example Response
{
  "pred_label": "DR",
  "pred_prob": 0.87,
  "latency_ms": 152,
  "report": "Predicted as Diabetic Retinopathy (confidence 0.87). Fundus shows microaneurysms..."
}

🏗️ Folder Structure
retina-backend/
│
├── app.py                 # FastAPI backend
├── requirements.txt
├── Dockerfile
├── models/
│   └── best_resnet34.pth  # model (optional local cache)
│
├── README.md
└── utils/                 # optional: xai/gradcam utils

🚀 Run Locally
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000


Or via Docker:

docker build -t retina-api .
docker run -d -p 8000:8000 \
  -e HF_REPO_ID=shrinanda/rfmid-10cls-resnet34 \
  -e MODEL_FILE=best_resnet34.pth \
  -e GEMINI_API_KEY=your_gemini_key_here \
  retina-api


Access API at → http://localhost:8000/docs

🧩 Integration with Frontend

Frontend (Flutter / React) →
Uploads an image to /report endpoint →
Displays disease prediction, confidence, and Gemini-generated report.

Example flow:

Upload image → FastAPI → ResNet-34 → Predicted class → Gemini → Report JSON → Display to user

🧠 Explainability (Planned)

Will integrate Grad-CAM, SHAP, and LIME for interpretability:

/xai/gradcam – Returns Grad-CAM overlay

/xai/shap – Returns SHAP force plots

/xai/lime – Returns local interpretable heatmaps

🧮 Performance Summary
Metric	Value
Validation Accuracy	0.81
Macro F1-Score	0.69
Best Model	best_resnet34.pth
Inference Time	120–200 ms per image (CPU)
🧑‍💻 Tech Stack
Layer	Technology
Model	PyTorch (ResNet-34)
API	FastAPI + Uvicorn
Model Hosting	Hugging Face Hub
Cloud Deployment	Oracle Cloud (Always-Free)
Report Generation	Google Gemini API
Containerization	Docker
Frontend	Flutter / React
🔮 Future Enhancements

Add Report PDF export via Gemini

Deploy to Render / Hugging Face Spaces
