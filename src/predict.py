import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("../model/medical_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Change image path here
img_path = r"C:\Users\nidhi\OneDrive\Desktop\Medical_Image_Classification\dataset\test\NORMAL\IM-0001-0001.jpeg"

img_array = preprocess_image(img_path)

prediction = model.predict(img_array)[0][0]

print("Raw Probability:", prediction)

if prediction > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")
