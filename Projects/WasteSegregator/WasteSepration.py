import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load trained Keras model
model_path = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Dry_Wet_classifier.h5"
try:
    model = load_model(model_path)
except Exception as e:
    print(f"[❌ ERROR] Failed to load model: {e}")
    exit()

# Define class labels
class_names = ['Dry Waste', 'Wet Waste']
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept prediction

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌ ERROR] Unable to open webcam.")
    exit()

print("[✅ INFO] Webcam opened. Press 'q' to quit.")

# Initialize time for FPS calculation
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️ WARNING] Failed to read frame. Exiting...")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Define bounding box in center of the frame
    box_size = 224
    x_center, y_center = width // 2, height // 2
    x1 = x_center - box_size // 2
    y1 = y_center - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Extract ROI from bounding box
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for model
    if roi.shape[0] == 224 and roi.shape[1] == 224:
        input_image = roi.astype("float32") / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Predict
        predictions = model.predict(input_image, verbose=0)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]

        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{class_names[class_id]} ({confidence * 100:.2f}%)"
        else:
            label = "Uncertain"
    else:
        label = "ROI Error"

    # Draw bounding box and prediction label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display the frame
    cv2.imshow('Garbage Classifier (ROI Mode)', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[👋 INFO] Exiting application.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
