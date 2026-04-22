#1
import cv2
from ultralytics import YOLO
import time
print(" Script started")

def test_yolo_person_detection():
    print("Loading YOLOv8 model...")
    
    # This will auto-download the model (~6MB)
    model = YOLO('yolov8n.pt')
    
    print(" Model loaded\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" Cannot access webcam")
        return

    print("Starting detection... Press 'q' to quit")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run detection (class 0 = person)
        results = model(frame, classes=[0], verbose=False)

        # Draw results
        annotated_frame = results[0].plot()

        # FPS calculation
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Person Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Done")

if __name__ == "__main__":
    test_yolo_person_detection()

    