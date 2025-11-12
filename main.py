import os
from io import BytesIO
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Banana Ripeness Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RIPENESS_LABELS: List[str] = [
    "Green",        # unripe
    "Green-Yellow", # early ripening
    "Yellow",       # ripe
    "Yellow-Brown", # late ripe / spotted
    "Brown"         # overripe
]


def load_image_to_array(file_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    try:
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img = img.resize(target_size)
        arr = np.asarray(img).astype("float32") / 255.0
        # Simple normalization to [0,1]
        return arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


def simple_color_based_model(arr: np.ndarray) -> List[float]:
    # Heuristic model: use average pixel color to guess ripeness
    # Green bananas have higher G channel, ripe more balanced with higher R, overripe darker (lower brightness)
    h, w, c = arr.shape
    avg = arr.reshape(-1, c).mean(axis=0)  # R,G,B
    brightness = arr.mean()

    r, g, b = avg[0], avg[1], avg[2]

    scores = np.zeros(5, dtype=np.float32)

    # Green: high G relative to R
    scores[0] = max(0.0, g - r + 0.3)

    # Green-Yellow: G still slightly higher but approaching balance
    scores[1] = max(0.0, (g + r) / 2 - abs(g - r) - 0.1)

    # Yellow: R slightly higher than G, high brightness
    scores[2] = max(0.0, r - g + brightness - 0.3)

    # Yellow-Brown: similar to yellow but lower brightness
    scores[3] = max(0.0, (r - g + 0.2) * (1.1 - brightness))

    # Brown: low brightness dominates
    scores[4] = max(0.0, 1.2 - 1.5 * brightness)

    # Softmax-like normalization
    if scores.sum() == 0:
        scores += 1e-3
    probs = (scores / scores.sum()).tolist()
    return probs


@app.get("/")
def root():
    return {"message": "Banana Ripeness Classifier API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Note: In a production-grade version, we'd load a trained ML model. Here we ship a heuristic
    # so the app works immediately without heavy training.
    content = await file.read()
    arr = load_image_to_array(content)
    probs = simple_color_based_model(arr)
    best_idx = int(np.argmax(probs))
    return {
        "label": RIPENESS_LABELS[best_idx],
        "probabilities": [
            {"label": RIPENESS_LABELS[i], "prob": float(p)} for i, p in enumerate(probs)
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
