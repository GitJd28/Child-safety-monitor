# 🛡️ FusionSight – AI Child Safety Monitoring System

FusionSight is a **real-time AI-powered child safety monitoring system** that analyzes live video feeds to detect potentially unsafe situations using computer vision and deep learning.

Unlike traditional baby monitors or CCTV systems, FusionSight understands context — not just detection.

---

## 🚀 What It Does

✔ Detects **who** is in the room (child vs adult)  
✔ Identifies **what** they are doing (falling, climbing, unsafe posture)  
✔ Recognizes **known vs unknown faces**  
✔ Combines all signals to decide **whether an alert is actually needed**  

All processing is done **on-device**, ensuring privacy and zero additional hardware cost.

---

## 🧠 How It Works (4-Layer Pipeline)

1️⃣ **Person Detection** – YOLOv8  
2️⃣ **Pose Estimation** – YOLOv8-Pose (17 keypoints)  
3️⃣ **Face Recognition** – Haar Cascade + LBPH  
4️⃣ **Context Engine** – Rule-based decision system for smart alerts  

---

## 🛠 Tech Stack

- Python  
- YOLOv8  
- OpenCV  
- NumPy  
- Haar Cascade & LBPH  

---

## 💡 Key Highlights

- ✅ Real-time monitoring on CPU  
- ✅ Context-aware alerts (not just motion detection)  
- ✅ Face enrollment system  
- ✅ Modular architecture for future AI integrations  
- ✅ Designed for homes, schools & childcare environments  

---

## 🔮 Future Scope

- Audio analysis (YAMNet)  
- SMS alerts (Twilio)  
- Edge deployment (Raspberry Pi)  
- Mobile application  

---

**Built to move from simple detection → intelligent safety awareness.**
