from flask import Flask, render_template, request, send_from_directory
from tensorflow.saved_model import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
# model = load('model_5class_v2')
model = load_model('model_5class_v2')

class_dict = {0: 'N dan K', 1: 'N, K, dan Mg', 2: 'N', 3: 'N, P, dan K', 4: 'N, P, K, dan Mg'}

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256), color_mode='rgba')
    print(loaded_img.size)
    img_array = img_to_array(loaded_img).astype(np.float32) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.argmax(model.predict(img_array))
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)