from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import uvicorn
import io
from PIL import Image
import logging

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS (Required for Streamlit to call FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define file paths
MODEL_PATH = "plant_disease_model.h5"
CSV_PATH = "class_data.csv"

def load_model(model_path):
    """Load and return the trained model."""
    try:
        logger.info("Loading model...")
        model = tf.keras.models.load_model(model_path)
        logger.info("✅ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load the trained model
model = load_model(MODEL_PATH)

def load_disease_info(csv_path):
    """Load disease information from CSV file."""
    try:
        logger.info("Loading disease information CSV...")
        disease_info = pd.read_csv(csv_path)
        logger.info("✅ CSV loaded successfully!")
        return disease_info
    except Exception as e:
        logger.error(f"❌ Error loading CSV: {e}")
        return None

# ✅ Load the disease information CSV
disease_info = load_disease_info(CSV_PATH)

def preprocess_image(img):
    """Preprocess image for model prediction."""
    img_size = (256, 256)  # Match training size
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(img):
    if model is None or disease_info is None:
        return {"error": "Model or CSV data not loaded!"}

    img_size = (256, 256)
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Get the class index
    logger.info(f"Predicted class index: {class_index}")

    if class_index < len(disease_info):
        disease_data = disease_info.iloc[class_index].to_dict()
        logger.info(f"Disease data: {disease_data}")
        return {"status": "success", "disease_info": disease_data}
    else:
        return {"error": "Predicted class index out of range!"}

# ✅ API Route to handle image uploads
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = predict_disease(img)
        return result
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        return {"error": str(e)}

# ✅ Run the backend
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)