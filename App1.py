from __future__ import division, print_function
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow import keras
import tensorflow as tf

from skimage.transform import resize

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import h5py

App1 = Flask(__name__)

Model1 = load_model(r'templates\Garbage1.h5')

@App1.route ('/',methods=["GET"])
def index():
    return render_template('index.html')
@App1.route ('/prediction',methods=['POST','GET'])
# route which will take you to the prediction page
def prediction():
    return render_template('prediction.html')


@App1.route('/result',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath,'uploads',f.filename)
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        preds = model.predict_classes(x)
        index =['cardboard','glass','metal','paper','plastic','trash']
        text = "The Predicted Garbage is : "+str(index[preds[0]])
        
        
        return text


if __name__=='__main__':
    App1.run(debug=True,threaded=False)