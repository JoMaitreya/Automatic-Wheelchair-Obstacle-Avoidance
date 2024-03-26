import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('synthetic_model.h5')

# Access the camera stream
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    # Resize, normalize, and perform any other preprocessing steps required by your model
    resized_frame = cv2.resize(frame, (224, 224))
    preprocessed_frame = resized_frame / 255.0  # Normalize pixel values

    # Perform inference
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0][0]

    # Display the frame with prediction
    if prediction < 0.5:
        cv2.putText(frame, 'No Obstacle', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Obstacle', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
