import os
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)


model=load_model('model.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    @app.route('/')
    def hello():
        return render_template("index.html")

    @app.route('/', methods=['POST'])
    def marks():
        if request.method == 'POST':
            f = request.files['userfile']
            path = "./static/{}".format(f.filename)
            f.save(path)
            preds = model_predict(path, model)
            result = preds
            return result

        return render_template("index.html")

    if __name__ == '__main__':
        app.run(debug=True)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Cat"

    else:
        preds = "Dog"

    return preds
