import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('emotion_model.keras')

# Class labels in the same order as during training
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.2)

# Start capturing from the default webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box in normalized coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Ensure the bounding box is within frame boundaries
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Extract the region of interest (the face) and preprocess
            face_roi = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face.astype('float32') / 255.0
            face_input = np.expand_dims(np.expand_dims(normalized_face, axis=-1), axis=0)  # Shape: (1, 48, 48, 1)

            # Run prediction
            prediction = model.predict(face_input, verbose=0)
            confidence = prediction.max()

            # Apply confidence threshold
            if confidence > 0.6:
                label = class_labels[np.argmax(prediction)]
            else:
                label = "Uncertain"

            # Display the predicted label above the bounding box
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

    # Show the annotated frame
    cv2.imshow('Real-Time Emotion Detection (MediaPipe)', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

