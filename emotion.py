
#!/usr/bin/env python3
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 0) Where your trained data folders live
train_dir = 'data/train'

# 1) List & lock in your labels from the exact subfolders
labels = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])
print("→ Using labels (in this order):", labels)

# 2) Load your final model
model = load_model('emotion_model4.keras')

# 3) MediaPipe face detector
face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.2
)

# 4) EMA smoothing state
ema_pred = None
alpha    = 0.4

# 5) Video capture + target face size
cap  = cv2.VideoCapture(0)
size = (224, 224)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- detect faces ---
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        # take the largest face
        areas = [
            d.location_data.relative_bounding_box.width *
            d.location_data.relative_bounding_box.height
            for d in results.detections
        ]
        d  = results.detections[np.argmax(areas)]
        bb = d.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1, y1 = int(bb.xmin * w), int(bb.ymin * h)
        x2, y2 = x1 + int(bb.width * w), y1 + int(bb.height * h)

        roi = frame[max(0,y1):y2, max(0,x1):x2]
        if roi.size:
            # --- preprocess exactly as at training ---
            face = cv2.resize(roi, size)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            inp  = preprocess_input(face.astype('float32'))
            inp  = np.expand_dims(inp, axis=0)  # shape (1,224,224,3)

            # --- predict + EMA smoothing ---
            p        = model.predict(inp, verbose=0)[0]
            ema_pred = p if ema_pred is None else alpha * p + (1 - alpha) * ema_pred

            # --- pick the highest-prob emotion ---
            label = labels[np.argmax(ema_pred)]

            # --- draw box & label ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255,255,255), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

