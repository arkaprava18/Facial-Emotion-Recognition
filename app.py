from flask import Flask, render_template, jsonify
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotions')
def detect_emotions():
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    emotion_counts = {label: 0 for label in labels_dict.values()}
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        emotion_counts[labels_dict[label]] += 1
    video.release()
    return jsonify(emotion_counts)

if __name__ == '__main__':
    app.run(debug=True)
