# 🛡️ Child Safety Monitoring System

An AI-powered real-time surveillance and monitoring system designed to enhance child safety using computer vision and deep learning techniques.

The system analyzes live CCTV/video feeds to automatically detect potentially unsafe situations through:

- Human Detection
- Pose Analysis
- Face Recognition
- Real-Time Monitoring

This project is aimed at improving safety monitoring in:

- Homes
- Schools
- Hospitals
- Childcare environments

---

# 🚀 Features

- ✅ Real-time person detection using YOLOv8
- ✅ Human pose estimation and posture analysis
- ✅ Face enrollment and recognition system
- ✅ Live webcam/video stream processing
- ✅ Personalized face database generation
- ✅ Detection pipeline for identifying known individuals
- ✅ Modular architecture for future AI integrations

---

# 🧠 Tech Stack

## AI / Computer Vision

- Python
- OpenCV
- YOLOv8
- OpenCV Face Recognizer
- NumPy

## Development Tools

- Git & GitHub
- Virtual Environment (venv)

---

# 📁 Project Structure

```bash
child-safety-monitor/
│
├── src/
│   ├── detection/
│   │   ├── opencv_face_recognizer.py
│   │   ├── pose_analyzer.py
│   │   ├── person_classifier.py
│   │   ├── test_pose.py
│   │   └── test_yolo.py
│   │
│   └── pipeline.py
│
├── scripts/
│   ├── prepare_face_database.py
│   ├── enroll_your_face.py
│   └── download_datasets.py
│
├── data/
├── outputs/
├── checkpoints/
└── models/
