# Low-Cost Multispectral Edge AI Monitoring Platform

A low-cost environmental monitoring platform combining multispectral imaging, deep learning, and full-stack deployment for vegetation analysis, crop–weed segmentation, and methane plume detection.

## Overview

Commercial multispectral monitoring systems are often expensive and inaccessible. This project presents a low-cost alternative by integrating affordable hardware, deep learning models, and a web-based monitoring application for environmental analysis.

The platform consists of:

- **Low-cost multispectral imaging hardware** for NDVI-based vegetation analysis
- **Deep learning pipelines** for crop–weed segmentation and methane plume detection
- **Full-stack deployment** for inference, visualization, and environmental monitoring

---

## Features

### Multispectral Imaging
- Raspberry Pi 5-based multispectral imaging system
- RGB + NoIR camera configuration
- 720nm IR filter for near-infrared capture
- ORB-based RGB–NIR image alignment
- NDVI heatmap generation for vegetation analysis

### AI / Computer Vision
- Crop–weed semantic segmentation using multispectral UAV imagery
- Methane plume detection using Sentinel-2 satellite imagery
- ONNX Runtime optimized inference
- Image preprocessing and inference pipelines

### Web Application
- FastAPI backend
- React frontend
- Interactive visualization dashboard
- Environmental monitoring workflows
- Voice-assisted interaction support

---

## System Architecture

```text
Low-Cost Multispectral Edge AI Monitoring Platform
│
├── Hardware Layer
│   ├── Raspberry Pi 5
│   ├── RGB Camera
│   ├── NoIR Camera + 720nm Filter
│   ├── Image Capture
│   └── NDVI Generation
│
├── AI Layer
│   ├── Crop-Weed Segmentation Model
│   ├── Methane Detection Model
│   ├── ONNX Runtime Inference
│   └── Preprocessing Pipeline
│
└── Application Layer
    ├── FastAPI Backend
    ├── React Frontend
    ├── Visualization Dashboard
    └── Environmental Monitoring Interface
```

---

## Tech Stack

### Languages
- Python
- JavaScript

### Machine Learning / Computer Vision
- PyTorch
- ONNX Runtime
- OpenCV
- NumPy

### Backend
- FastAPI
- Uvicorn

### Frontend
- React
- Vite

### Hardware
- Raspberry Pi 5
- Raspberry Pi Camera Module 2
- Raspberry Pi NoIR Camera Module 2

### Tools
- Git
- VS Code
- Google Colab
- Fusion 360

---

## Deep Learning Models

### Crop–Weed Segmentation

Deep learning pipeline for multispectral agricultural image segmentation.

**Dataset:** WeedsGalore  
**Task:** Semantic segmentation  
**Classes:** Crop / Weed / Background  
**Performance:** **83.86% mIoU**

---

### Methane Plume Detection

Satellite-based methane plume segmentation using remote sensing imagery.

**Dataset:** Sentinel-2 imagery  
**Task:** Binary segmentation  
**Performance:** **0.92 scene-level recall**

---

## NDVI Pipeline

1. Capture RGB image
2. Capture NIR image
3. Extract ORB features
4. Perform image registration
5. Apply homography alignment
6. Compute NDVI
7. Generate heatmap visualization

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Low-Cost-Multispectral-Edge-AI-Monitoring-Platform.git
cd Low-Cost-Multispectral-Edge-AI-Monitoring-Platform
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Project Structure

```text
Low-Cost-Multispectral-Edge-AI-Monitoring-Platform/
│
├── backend/
│   ├── models/
│   ├── preprocessing/
│   ├── api/
│   └── app.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── ndvi/
│   ├── capture.py
│   ├── alignment.py
│   └── ndvi.py
│
├── trained_models/
│   ├── crop_weed_model.onnx
│   └── methane_model.onnx
│
└── README.md
```

---

## Applications

- Precision agriculture
- Crop health monitoring
- Weed detection
- Environmental analytics
- Methane emission monitoring
- Low-cost remote sensing research

---

## Results

| Task | Performance |
|------|-------------|
| Crop–Weed Segmentation | 83.86% mIoU |
| Methane Detection | 0.92 Recall |

---

## Contributors

**Sanjana Suresh**

---

## License

Academic / Research Project
