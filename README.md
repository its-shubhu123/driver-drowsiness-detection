# Driver Drowsiness Detection System

A real-time AI-based system that detects driver drowsiness using computer vision and deep learning techniques. This project helps prevent accidents by alerting the driver when signs of fatigue (such as eye closure or yawning) are detected.

---

##  Features

- Eye state detection (Open/Closed)
- Yawning detection
- Real-time monitoring using webcam
- Alarm system for drowsiness alert
- Deep learning-based prediction
- Optimized for low-resource systems

---

## Tech Stack

- Programming Language: Python  
- Libraries:
  - OpenCV  
  - MediaPipe  
  - TensorFlow / Keras  
  - NumPy  

- Models Used:
  - MobileNetV2  
  - LSTM  

---

## Project Structure

Driver-Drowsiness-Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   └── model.h5
│
├── src/
│   ├── detect_drowsiness.py
│   └── utils.py
│
├── alarm/
│   └── alarm.wav
│
├── requirements.txt
├── main.py
└── README.md

---

## Installation

1. Clone the repository:
git clone https://github.com/its-shubhu123/driver-drowsiness-detection.git

2. Navigate to project directory:
cd driver-drowsiness-detection

3. Create virtual environment (optional):
python -m venv venv

Activate environment:
- Windows: venv\Scripts\activate
- Linux/Mac: source venv/bin/activate

4. Install dependencies:
pip install -r requirements.txt

---

## Usage

Run the project:
python main.py

- Webcam will start automatically
- System will detect eye closure and yawning
- Alarm will trigger if drowsiness is detected

---

## 🧠 How It Works

1. Face detection using MediaPipe  
2. Facial landmark extraction (eyes and mouth)  
3. Feature calculation:
   - Eye Aspect Ratio (EAR)
   - Mouth Opening Ratio  
4. Deep learning model prediction using MobileNetV2 + LSTM  
5. Alert generation if drowsiness is detected  

---

## 📊 Dataset


---

## 🚀 Future Improvements

- Add head pose detection  
- Deploy as mobile app  
- Integrate with IoT-based car systems  
- Improve accuracy with larger dataset  
- Add night vision capability  
