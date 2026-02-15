from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/medical_model.h5")

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    confidence_text = ""
    image_file = ""

    if request.method == "POST":
        file = request.files["file"]

        if file:
            # Create static folder if not exists
            os.makedirs("static", exist_ok=True)

            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            image_file = file.filename

            # Preprocess image
            img_array = preprocess_image(file_path)

            # Predict
            prediction = model.predict(img_array)[0][0]
            confidence = round(float(prediction) * 100, 2)

            if prediction > 0.5:
                prediction_text = "PNEUMONIA"
                confidence_text = f"Confidence: {confidence}%"
            else:
                prediction_text = "NORMAL"
                confidence_text = f"Confidence: {100 - confidence}%"

    return render_template("index.html",
                           prediction=prediction_text,
                           confidence=confidence_text,
                           image_file=image_file)

if __name__ == "__main__":
    app.run(debug=True)
