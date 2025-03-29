import cv2
import numpy as np
import winsound  # For Windows alarm sound
from tensorflow.keras.models import load_model
from playsound import playsound  # For custom sound

# Load the trained model
model = load_model("saved_model/drowsiness_model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

CATEGORIES = ["Closed_Eyes", "Open_Eyes"]


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64)).reshape(-1, 64, 64, 1) / 255.0  # Resize & Normalize

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = CATEGORIES[class_index]

    # Display the prediction on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Play alarm sound if drowsiness is detected
    if label == "Closed_Eyes":
        print("⚠️ ALERT! Driver is Drowsy! ⚠️")
        # winsound.Beep(1000, 500)  # Windows built-in beep sound (uncomment if needed)
        playsound("alarm.wav")  # Custom alarm sound

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
