from flask import Flask, render_template, request
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('test.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        img_path = 'uploads/' + uploaded_file.filename
        uploaded_file.save(img_path)

        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)

        classes = ['anya', 'aqua', 'ayaka', 'boa', 'charlotte','damian','dazai','gojo','jett','kafka','killjoy','kugisaki','loid',
                   'luffy','midoriya','minato','misa','mitsuri','nico','nier','niera2','raiden','ruby','sakura','todoroki','tokisaki',
                   'uraraka','wanderer','yor','yuri','zerotwo']  # Replace with your actual class names
        predicted_class = classes[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
