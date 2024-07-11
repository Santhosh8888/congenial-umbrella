import re
import sys
import numpy as np
import os
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image

# Loading the model
model = load_model("model.h5")

app = Flask(__name__)

# default home page or route
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/prediction.html")
def prediction():
    return render_template('prediction.html')

@app.route("/index.html")
def home():
    return render_template('index.html')

@app.route("/result", methods=["GET", "POST"])
def res():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Resize the image to (224, 224)
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        prediction = np.argmax(model.predict(x), axis=1)
        index = ["trash", "glass", "metal", "paper", "cardboard", "plastic"]
        result = index[prediction[0]]

        return render_template('prediction.html', prediction=result)

""" Running our application """
if __name__ == "__main__":
    app.run(debug=True, port=5000)
