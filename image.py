import cv2
import numpy as np
from keras.models import load_model
import os

# Check if model files exist
faceProto = "models/deploy.prototxt"
faceModel = "models/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.isfile(faceProto):
    print(f"Error: {faceProto} does not exist.")
if not os.path.isfile(faceModel):
    print(f"Error: {faceModel} does not exist.")

# Load the DNN-based face detector model
faceNet = cv2.dnn.readNet(faceProto, faceModel)

# Load the pre-trained emotion recognition model
model = load_model('model_file_30epochs.h5')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the static image
image_path = 'image\im.jpg'
print(f"Current Working Directory: {os.getcwd()}")
print(f"Image Path: {image_path}")

image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve the image contrast (optional but can help)
    gray_image = cv2.equalizeHist(gray_image)

    # Prepare the image for the DNN face detector
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

    # Detect faces
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Print the total number of detected faces
    total_faces = 0
    emotion_counts = {label: 0 for label in labels_dict.values()}

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            total_faces += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Extract the face region
            sub_face_img = gray_image[y:y1, x:x1]

            # Resize and normalize the face image
            if sub_face_img.size == 0:
                continue
            resized = cv2.resize(sub_face_img, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))

            # Predict emotion label
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            # Increment emotion count
            emotion_counts[labels_dict[label]] += 1

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)

            # Display emotion label
            cv2.putText(image, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display total number of detected faces
    cv2.putText(image, f"Total Faces: {total_faces}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display emotion counts
    y_pos = 50  # Adjust y position to avoid overlap with the total faces count
    for label, count in emotion_counts.items():
        cv2.putText(image, f"{label}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30

    # Display the processed image
    window_name = "Emotion Detection on Static Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, image.shape[1], image.shape[0])  # Adjust window size to image size

    # Wait until a key is pressed and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
