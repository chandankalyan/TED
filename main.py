# main.py
from fastapi import FastAPI, File, UploadFile
import shutil
import tempfile
import os
import tensorflow as tf
from predictor import preprocess_image, decode_predictions
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile):
    # Create a temporary directory to save the uploaded image
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = os.path.join(temp_dir, file.filename)

        # Save the uploaded image to the temporary directory
        with open(image_path, "wb") as image_file:
            shutil.copyfileobj(file.file, image_file)

        # Preprocess the image (use your existing code)
        test_img = preprocess_image(image_path)

        # Reshape the image
        test_img = tf.reshape(test_img, (1, 128, 32, 1))

        # Load the model (use your existing code)
        model = tf.keras.models.load_model('model.h5')

        # Make predictions (use your existing code)
        prediction = model.predict(test_img)

        # Decode the prediction (use your existing code)
        decoded = decode_predictions(prediction)

        # Return the prediction as JSON
        return {"prediction": decoded[0]}



app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)
