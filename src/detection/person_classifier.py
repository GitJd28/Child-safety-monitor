#2
import cv2
from ultralytics import YOLO

class PersonClassifier:
    def __init__(self, child_threshold=0.5):
        self.model = YOLO('yolov8n.pt')
        self.child_threshold = child_threshold

    def detect_and_classify(self, frame):
        results = self.model(frame, classes=[0], verbose=False)

        frame_height = frame.shape[0]
        persons = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            person_height = y2 - y1
            height_ratio = person_height / frame_height

            if height_ratio < self.child_threshold:
                person_type = "child"
            else:
                person_type = "adult"

            persons.append({
                "bbox": (x1, y1, x2, y2),
                "type": person_type,
                "confidence": conf,
                "height_ratio": height_ratio
            })

        return persons

    def draw(self, frame, persons):
        for p in persons:
            x1, y1, x2, y2 = p["bbox"]

            color = (0, 165, 255) if p["type"] == "child" else (0, 255, 0)

            label = f"{p['type']} ({p['height_ratio']:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame


def run():
    clf = PersonClassifier()

    cap = cv2.VideoCapture(0)

    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = clf.detect_and_classify(frame)
        frame = clf.draw(frame, persons)

        # Count
        children = sum(1 for p in persons if p["type"] == "child")
        adults = sum(1 for p in persons if p["type"] == "adult")

        cv2.putText(frame, f"Children: {children} | Adults: {adults}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Child vs Adult", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()