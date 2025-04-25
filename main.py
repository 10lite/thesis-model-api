from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="Durian Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic model for response
class DurianClassification(BaseModel):
    photo: str  # Base64-encoded image
    class_name: str  # "puyat" or "nonpuyat"
    confidence: float  # Confidence score

@app.get("/")
def read_root():
    return {"message": "Welcome to Durian Classification API!"}

# Load YOLO model at startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO("durian_variety_classifier.pt")  # Trained model for durian classification
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load YOLO model: {e}",
        )

@app.post("/classify/", response_model=DurianClassification)
async def classify_durian(input: UploadFile) -> DurianClassification:
    try:
        # Load image
        image_bytes = await input.read()
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        
        # Preprocess image
        image_array = np.array(image)
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # YOLO model inference
        result = model(opencv_image, conf=0.80)

        # Extract bounding boxes, classes and confidence scores
        boxes = result[0].boxes.xyxy.cpu().numpy()
        confidences = result[0].boxes.conf.cpu().numpy()
        class_ids = result[0].boxes.cls.cpu().numpy()
        
        # Classes
        class_names = ["puyat", "nonpuyat"]
        class_colors = {
            "puyat": (0, 255, 0),      # Green
            "nonpuyat": (0, 0, 255)    # Red
        }

        # If no durians detected
        if len(boxes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No durians were detected in the image",
            )

        # Count detections and calculate average confidence
        class_counter = {name: 0 for name in class_names}
        total_confidence = 0.0
        for conf, cls_id in zip(confidences, class_ids):
            class_name = class_names[int(cls_id)]
            class_counter[class_name] += 1
            total_confidence += conf
        average_conf = float(total_confidence / len(confidences))

        # If only one instance, return class name
        if len(class_counter) == 1:
            class_name = next(iter(class_counter.keys()))
            class_summary = f"{class_counter[class_name]} {class_name}s"
        else: # Else, summarize the counts
            class_summary = ", ".join(f"{count} {name}s" for name, count in class_counter.items() if count > 0)
        
        # Dotted bounding box
        def draw_dotted_rectangle(img, pt1, pt2, color, gap=5):
            x1, y1 = pt1
            x2, y2 = pt2
            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y1), (x + gap, y1), color, thickness=2)
                cv2.line(img, (x, y2), (x + gap, y2), color, thickness=2)
            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x1, y), (x1, y + gap), color, thickness=2)
                cv2.line(img, (x2, y), (x2, y + gap), color, thickness=2)

        # Draw each bounding box and label
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cls = class_names[int(cls_id)]
            color = class_colors[cls]

            draw_dotted_rectangle(image_array, (x1, y1), (x2, y2), color)

            label = f"{cls} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(image_array,
                          (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_size[0] + 4, text_y + 4),
                          color, thickness=-1)
            cv2.putText(image_array, label, (text_x, text_y), font, font_scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Convert back to JPEG and encode
        image_pil = Image.fromarray(image_array)
        image_stream = io.BytesIO()
        image_pil.save(image_stream, format="JPEG", quality=95, optimize=True)
        encoded_image = base64.b64encode(image_stream.getvalue()).decode()

        return DurianClassification(
            photo=encoded_image,
            class_name=class_summary,
            confidence=average_conf
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {e}",
        )