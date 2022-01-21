from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, send
from anpr import plateRecognizor, paths
from PIL import Image
import base64
import numpy as np
import io
import os


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    print(data)

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars428.png')
recognizor = plateRecognizor(IMAGE_PATH, True)

@app.route("/detect", methods=['POST'])
def detect():
    dataURL = request.json['data']
    print('got dataURL')
    img_base64 = dataURL.split(',')[1]
    binary = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(binary))
    img = img.convert('RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img = np.array(img)
    # img = cv2.flip(img, 1)
    isDetected, detections, img_np_with_detections = recognizor.detectPlate(img)
    text_array = []
    img_plate_b64 = ''
    if isDetected:
        text_array, img_plate_b64 = recognizor.detectOCR(detections, img_np_with_detections)
    result = {"detected": isDetected, "img_base64": img_plate_b64, "text": text_array}
    return jsonify(result)

# if __name__ == '__main__':
    # context = ('local.crt', 'local.key') # certificate and key files
    # app.run('0.0.0.0', debug=True, port=8100, ssl_context=context)
    # app.run('0.0.0.0', ssl_context='adhoc')
    # app.run('0.0.0.0')
