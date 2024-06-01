import cv2
import numpy as np
from keras.models import load_model
import os

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Load the Haar cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the static image
image_path = 'a.jpg'
print(f"Current Working Directory: {os.getcwd()}")
print(f"Image Path: {image_path}")

image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceDetect.detectMultiScale(gray_image, 1.3, 5)

    # Print the total number of detected faces
    print(f"Total number of detected faces: {len(faces)}")

    # Initialize emotion counts
    emotion_counts = {label: 0 for label in labels_dict.values()}

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Extract the face region
        sub_face_img = gray_image[y:y+h, x:x+w]

        # Resize and normalize the face image
        resized = cv2.resize(sub_face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))

        # Predict emotion label
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Increment emotion count
        emotion_counts[labels_dict[label]] += 1

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display emotion label
        cv2.putText(image, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display emotion counts
    y_pos = 30
    for label, count in emotion_counts.items():
        cv2.putText(image, f"{label}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30

    # Display the processed image
    cv2.imshow("Emotion Detection on Static Image", image)

    # Wait until a key is pressed and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
