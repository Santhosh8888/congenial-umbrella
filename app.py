import re
import sys
import numpy as np
import os
from flask import Flask, app, request, render_template, url_for, redirect
from keras import models
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat

# Loading the model
# model = load_model("C:/Users/91902/Downloads/garbage_new.h5")
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
        basepath = os.path.dirname(__file__)  # getting the current path i.e where app.py is present
        # print("current path",basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)  # from anywhere in the system we can give image but we want that image later  to process so we are saving it to uploads folder for reusing
        # print("upload folder is",filepath)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)  # img to array
        x = np.expand_dims(x,axis=0)  # used for adding one more dimension
        # print(x)
        prediction = np.argmax(model.predict(x),axis=1)  # instead of predict_classes(x) we can use predict(X) ---->predict_classes(x) gave error
        # print("prediction is ",prediction)
        index = ["cardboard","glass","metal","paper","plastic","trash"]
        result = str(index[prediction[0]])
        result
        return render_template('prediction.html', prediction=result)


""" Running our application """
if __name__ == "__main__":
    app.run(debug=True, port=5000)