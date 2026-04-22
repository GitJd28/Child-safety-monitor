#3
# src/detection/test_pose.py
import cv2
from ultralytics import YOLO
import numpy as np

def test_pose_estimation():
    """Test YOLOv8 pose estimation"""
    
    print("Loading YOLOv8-Pose model...")
    # This downloads yolov8n-pose.pt (~6MB)
    model = YOLO('yolov8n-pose.pt')
    print("✅ Model loaded\n")
    
    cap = cv2.VideoCapture(0)
    print("Pose Estimation Test")
    print("Move around to see skeleton tracking")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation
        results = model(frame, verbose=False)
        
        # YOLOv8-pose automatically plots skeleton
        annotated = results[0].plot()
        
        # Count people detected
        num_people = len(results[0].boxes) if results[0].boxes is not None else 0
        
        cv2.putText(annotated, f"People detected: {num_people}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Pose Estimation', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test complete")


if __name__ == "__main__":
    test_pose_estimation()