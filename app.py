import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
# from auth import auth

app = Flask(__name__)
app.config['ALLOWED_EXTENSION'] = set({'png', 'jpg', 'jpeg'})
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

def allowed_file(filename):
    return '.' in filename and \
        filename.split('.', 1)[1] in app.config['ALLOWED_EXTENSION']

# Load the trained model
model = load_model('new_anime_classification2.h5', compile=False)
with open('labels.txt', 'r') as file:
    labels = file.read().splitlines()

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Welcome to Anime Classification API",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            # Save input image
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Pre-processing input image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((128, 128))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0) # (128, 128, 3)
            normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
            data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
            data[0] = normalized_img_array

            # Predicting the image
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = labels[index]
            class_name = class_name[2:]
            confidence = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success Predicting",
                },
                "data": {
                    "anime_classification": class_name,
                    "confidence": float(confidence)
                },
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad Request",
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method Not Allowed",
            },
            "data": None,
        }), 405

if __name__ == '__main__':
    app.run()

#@app.route('/predict', methods=['POST'])
#def predict():
#    uploaded_file = request.files['file']
#    if uploaded_file.filename != '':
#        img_path = 'uploads/' + uploaded_file.filename
#        uploaded_file.save(img_path)
#
#        img = image.load_img(img_path, target_size=(128, 128))
#        img_array = image.img_to_array(img)
#        img_array = np.expand_dims(img_array, axis=0) / 255.0

#        prediction = model.predict(img_array)

#        classes = ['Anya_Forger', 'Aquamarine_Hoshino', 'Ayaka_Genshin_Impact', 'Boa_Hancock', 'Charlotte_Genshin_Impact', 'Damian_Desmond', 'Dazai_Osamu_BSD', 'Ganyu_genshin', 'Gojo_Satoru', 'Jett_Valorant', 'Kafka_Honkai_Star_Rail', 'Keqing_genshin', 'Killjoy_Valorant', 'Kobo_kanaeru', 'Kugisaki_Nobara', 'Loid_Forger', 'Luffy_D_Monkey', 'Midoriya_Izuku', 'Minato_Aqua', 'Misa_Amane', 'Mitsuri_Kanroji', 'Nico_Robin', 'Nier_Automata_9S', 'Nier_Automata_A2', 'Raiden_Shogun_Genshin_Impact', 'Ruby_Hoshino', 'Sakura_Haruno', 'Todoroki_Shoto', 'Tokisaki_Kurumi', 'Uraraka_Ochako', 'Wanderer', 'Yor_Forger', 'Yuri_Briar', 'Zerotwo', 'amelia_watson', 'arima_kousei', 'fern', 'frieren', 'gawr_gura', 'hoshino_ai', 'hutao_genshin', 'kaori', 'mikasa', 'tanjiro', 'violet_evergarden', 'zeta_hololive']  # Replace with your actual class names
#        predicted_class = classes[np.argmax(prediction)]

#        return render_template('result.html', prediction=predicted_class, img_path=img_path)


# from flask import Flask, render_template, request
# import os
# from keras.models import load_model
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__, template_folder='templates')

# # Load the trained model
# model = load_model('anime_classification_va70_val01.h5')

# # Define your class labels based on folder names
# class_labels = os.listdir('datasets_character_anime')  # Adjust this based on your dataset path
# class_labels.sort()  # Ensure consistent order

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     uploaded_file = request.files['file']
#     if uploaded_file.filename != '':
#         img_path = 'uploads/' + uploaded_file.filename
#         uploaded_file.save(img_path)

#         # Extract class label from the folder name
#         folder_name = os.path.dirname(img_path)
#         class_label = os.path.basename(folder_name)

#         # Check if the class label is valid
#         if class_label not in class_labels:
#             return render_template('error.html', message='Invalid class label')

#         img = image.load_img(img_path, target_size=(128, 128))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0

#         prediction = model.predict(img_array)

#         predicted_class_index = np.argmax(prediction)
#         predicted_class = class_labels[predicted_class_index]

#         return render_template('result.html', prediction=predicted_class, img_path=img_path)

# if __name__ == '__main__':
#     app.run(debug=True)


