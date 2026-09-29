import io
import os
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Warehouse Box Detection System", version="2.5")

# Enable CORS for flexible local development and API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Discover available .pt models in the workspace
def get_available_models() -> List[str]:
    return [f.name for f in Path(".").glob("*.pt")]

MODEL_PATH = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
print(f"Loading YOLO model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names if hasattr(model, 'names') else {0: 'box'}
print(f"Model loaded successfully. Classes: {CLASS_NAMES}")


class FramePayload(BaseModel):
    image: str  # Base64 encoded image (data URI or raw base64)
    confidence: Optional[float] = 0.35


class ModelSwitchPayload(BaseModel):
    model_name: str


# ---------------------------------------------------------------------------
# UI Page & Model Metadata Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def read_index():
    html_path = Path("Upload_for_Detection.html")
    if html_path.exists():
        return FileResponse(html_path)
    return JSONResponse(status_code=404, content={"message": "Upload_for_Detection.html not found"})


@app.get("/example_input.jpg")
async def get_example_image():
    example_path = Path("example_input.jpg")
    if example_path.exists():
        return FileResponse(example_path)
    return JSONResponse(status_code=404, content={"message": "Example image not found"})


@app.get("/api/model_info")
async def get_model_info():
    """Returns currently loaded model metadata and list of available models."""
    global MODEL_PATH, CLASS_NAMES
    return {
        "active_model": MODEL_PATH,
        "classes": CLASS_NAMES,
        "class_list": list(CLASS_NAMES.values()) if isinstance(CLASS_NAMES, dict) else CLASS_NAMES,
        "available_models": get_available_models()
    }


@app.post("/api/switch_model")
async def switch_model(payload: ModelSwitchPayload):
    """Allows hot-swapping between any .pt model files found in directory."""
    global model, MODEL_PATH, CLASS_NAMES
    target = payload.model_name
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"Model file '{target}' not found.")
    
    try:
        model = YOLO(target)
        MODEL_PATH = target
        CLASS_NAMES = model.names if hasattr(model, 'names') else {0: 'box'}
        return {
            "success": True,
            "message": f"Switched model to {target}",
            "active_model": MODEL_PATH,
            "classes": CLASS_NAMES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


# ---------------------------------------------------------------------------
# High Performance API for Live Camera Feed & UI
# ---------------------------------------------------------------------------
@app.post("/api/detect_frame")
async def detect_frame(payload: FramePayload):
    """
    High-performance endpoint for live camera frame streaming via Base64.
    Dynamically handles any model classes (box, Regular_Box, Large_Box, etc.).
    """
    try:
        t_start = time.time()
        
        # Decode base64 image data
        image_data = payload.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
            
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        conf_threshold = max(0.05, min(0.99, payload.confidence or 0.35))
        
        # Run YOLO inference
        results = model.predict(source=img_cv, conf=conf_threshold, verbose=False)
        t_inference = (time.time() - t_start) * 1000

        boxes_data = []
        class_counter = Counter()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

                class_counter[cls_name] += 1

                boxes_data.append({
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return {
            "success": True,
            "inference_ms": round(t_inference, 1),
            "total_boxes": len(boxes_data),
            "class_counts": dict(class_counter),
            "boxes": boxes_data,
            "image_width": img_cv.shape[1],
            "image_height": img_cv.shape[0]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/detect_file")
async def detect_file(
    file: UploadFile = File(...),
    confidence: float = Form(0.35)
):
    """
    Endpoint for single image file detection (Photo Mode).
    Returns detection statistics, class breakdowns, and base64 rendered annotated image.
    """
    try:
        t_start = time.time()
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_cv is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        conf_threshold = max(0.05, min(0.99, confidence))
        results = model.predict(source=img_cv, conf=conf_threshold, verbose=False)
        t_inference = (time.time() - t_start) * 1000

        boxes_data = []
        class_counter = Counter()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for b in boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                cls_name = CLASS_NAMES.get(cls_id, f"Class_{cls_id}")

                class_counter[cls_name] += 1

                boxes_data.append({
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2),
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        # Generate annotated image
        annotated_cv = results[0].plot() if len(results) > 0 else img_cv
        _, buffer = cv2.imencode('.jpg', annotated_cv)
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "inference_ms": round(t_inference, 1),
            "total_boxes": len(boxes_data),
            "class_counts": dict(class_counter),
            "boxes": boxes_data,
            "annotated_image": annotated_base64,
            "image_width": img_cv.shape[1],
            "image_height": img_cv.shape[0]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Legacy Endpoints for Backwards Compatibility
# ---------------------------------------------------------------------------
@app.post("/YOLO_Box_Prediction_Website/")
def predict_uploaded_image_website(file: UploadFile):
    """Legacy endpoint that returns the annotated image file response."""
    try:
        file_name = file.filename or "uploaded_image.jpg"
        temp_path = Path("temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        results = model(str(temp_path))
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        output_path = Path("example_output.jpg")
        im.save(output_path)
        return FileResponse(output_path)
    except Exception as e:
        return {"message": str(e)}


@app.post("/YOLO_Box_Prediction_Service/")
async def predict_uploaded_image_service(file: UploadFile):
    """Legacy service endpoint."""
    try:
        temp_path = Path("temp_service_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        results = model(str(temp_path))
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        output_path = Path("example_output.jpg")
        im.save(output_path)

        file_metadata = {
            "file_name": str(output_path),
            "file_size": os.path.getsize(output_path)
        }
        return {"file_metadata": file_metadata, "file_data": FileResponse(output_path)}
    except Exception as e:
        return {"message": str(e)}