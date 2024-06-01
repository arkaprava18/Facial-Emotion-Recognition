import cv2
import numpy as np
from keras.models import load_model

#pre-trained model
model = load_model('model_file_30epochs.h5')

#video capture device
video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion face
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize emotion counts
emotion_counts = {label: 0 for label in labels_dict.values()}

while True:
    # Read frame from the video
    ret, frame = video.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    
    # Initialize face count
    face_count = 0
    
    for label in labels_dict.values():
        emotion_counts[label] = 0
    
    for x, y, w, h in faces:
        face_count += 1
        
        sub_face_img = gray[y:y+h, x:x+w]
        
        # Resize and normalize the face image
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        emotion_counts[labels_dict[label]] += 1
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display the number of faces detected
    cv2.putText(frame, f"Number of faces detected: {face_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display emotion counts
    y_pos = 50
    for label, count in emotion_counts.items():
        cv2.putText(frame, f"{label}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30
    
    # Display the frame
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
