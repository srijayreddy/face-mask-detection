import cv2
import numpy as np

# Load pre-trained face detection model
face_net = cv2.dnn.readNet('face_detector.prototxt', 'face_detector.caffemodel')

# Load pre-trained mask detection model
mask_net = cv2.dnn.readNet('face_mask_detector.model')

# Function to detect face masks
def detect_mask(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Check if the detection is a face and the confidence is high enough
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Use histogram equalization for better contrast
            face_gray = cv2.equalizeHist(face_gray)

            # Apply Gaussian blur to reduce noise
            face_blur = cv2.GaussianBlur(face_gray, (5, 5), 0)

            # Apply adaptive thresholding to extract mask area
            _, mask = cv2.threshold(face_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations to smooth the mask and fill holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            mask = cv2.dilate(mask, None, iterations=2)

            # Calculate the percentage of mask area
            total_pixels = mask.shape[0] * mask.shape[1]
            mask_pixels = cv2.countNonZero(mask)
            mask_ratio = mask_pixels / total_pixels

            # Define label and color based on mask ratio
            if mask_ratio > 0.2:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            # Draw the bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # Add label
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return frame

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = detect_mask(frame)

    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

