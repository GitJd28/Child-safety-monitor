# src/detection/opencv_face_recognizer.py
"""
Improved Face Recognizer using LBPH.

Improvements over original:
1. Proper unknown rejection (confidence-based)
2. Returns structured data (not just draws on frame)
3. Can be imported by other modules (pipeline integration)
4. Better confidence threshold tuning
5. Saves/loads trained model (no retraining every run)
"""

import cv2
import os
import numpy as np
import pickle


class SimpleFaceRecognizer:
    
    def __init__(self, dataset_path="data/face_dataset"):
        
        self.dataset_path = dataset_path
        self.model_path = "models/lbph_model.yml"
        self.labelmap_path = "models/label_map.pkl"
        
        # Haar cascade for face detection
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # LBPH recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Label map: {0: 'jd', 1: 'friend', ...}
        self.label_map = {}
        
        # Reverse map: {'jd': 0, 'friend': 1, ...}
        self.name_to_label = {}
        
        # Confidence threshold
        # LBPH: lower = more confident match
        # < 70  = known person (confident)
        # 70-85 = uncertain
        # > 85  = unknown / stranger
        self.confidence_threshold = 95
        
        self.is_trained = False
    
    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────
    
    def prepare_dataset(self):
        """
        Load all images from dataset_path.
        Detects faces and pairs them with labels.
        
        Returns:
            faces: list of grayscale face crops
            labels: list of integer labels
        """
        faces = []
        labels = []
        current_label = 0
        
        print("Preparing dataset...")
        print(f"  Dataset path: {os.path.abspath(self.dataset_path)}\n")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return [], []
        
        persons = sorted(os.listdir(self.dataset_path))
        
        for person_name in persons:
            person_path = os.path.join(self.dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            self.label_map[current_label] = person_name
            self.name_to_label[person_name] = current_label
            
            image_files = [
                f for f in os.listdir(person_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            person_faces = 0
            
            for img_name in image_files:
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                detected = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in detected:
                    face_crop = gray[y:y+h, x:x+w]
                    
                    # Resize to standard size for consistency
                    face_resized = cv2.resize(face_crop, (200, 200))
                    face_resized = cv2.equalizeHist(face_resized)

                    
                    faces.append(face_resized)
                    labels.append(current_label)
                    person_faces += 1
            
            status = "✅" if person_faces >= 15 else "⚠️ "
            print(f"  {status} {person_name}: {person_faces} faces loaded "
                  f"(from {len(image_files)} images)")
            
            if person_faces < 15:
                print(f"     ↳ Recommend at least 15 face crops. "
                      f"Add more photos to data/face_dataset/{person_name}/")
            
            current_label += 1
        
        print(f"\n  Total: {len(faces)} face samples, "
              f"{len(self.label_map)} persons")
        
        return faces, labels
    
    def train(self, force_retrain=False):
        """
        Train LBPH model.
        Saves model to disk so we don't retrain every run.
        
        Args:
            force_retrain: if True, retrain even if saved model exists
        """
        os.makedirs("models", exist_ok=True)
        
        # Load saved model if exists and not forcing retrain
        if (not force_retrain and 
            os.path.exists(self.model_path) and 
            os.path.exists(self.labelmap_path)):
            
            print("Loading saved model...")
            self.recognizer.read(self.model_path)
            
            with open(self.labelmap_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.label_map = saved_data['label_map']
                self.name_to_label = saved_data['name_to_label']
            
            print(f"✅ Model loaded: {list(self.label_map.values())}")
            self.is_trained = True
            return True
        
        # Train from scratch
        faces, labels = self.prepare_dataset()
        
        if len(faces) == 0:
            print("❌ No faces found in dataset. Cannot train.")
            return False
        
        # if len(set(labels)) < 2:

        #     print("❌ Need at least 2 different people in dataset.")
        #     print("   Add photos for more people.")
        #     return False
        
        if len(set(labels)) < 2:
            print("⚠️  Single person mode - anyone else = UNKNOWN")
            dummy = np.zeros((200, 200), dtype=np.uint8)
            faces.append(dummy)
            next_label = max(labels) + 1
            labels.append(next_label)
            self.label_map[next_label] = '__dummy__'
        
        print("\nTraining LBPH model...")
        self.recognizer.train(faces, np.array(labels))
        
        # Save model
        self.recognizer.save(self.model_path)
        
        with open(self.labelmap_path, 'wb') as f:
            pickle.dump({
                'label_map': self.label_map,
                'name_to_label': self.name_to_label
            }, f)
        
        print(f"✅ Training complete!")
        print(f"   Model saved: {self.model_path}")
        print(f"   Known persons: {list(self.label_map.values())}")
        
        self.is_trained = True
        return True
    
    # ─────────────────────────────────────────────
    # RECOGNITION (Core Function)
    # ─────────────────────────────────────────────
    
    def recognize_frame(self, frame):
        """
        Detect and recognize all faces in a frame.
        This is the function used by the pipeline.
        
        Args:
            frame: BGR image (numpy array) from webcam
            
        Returns:
            list of dicts, one per detected face:
            [
                {
                    'name': 'jd',           # or 'UNKNOWN'
                    'is_known': True,        # False if unknown
                    'confidence': 45.2,      # LBPH confidence (lower=better)
                    'bbox': (x, y, w, h),   # face location in frame
                    'label': 0              # integer label (-1 if unknown)
                },
                ...
            ]
        """
        if not self.is_trained:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_detected = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces_detected:
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (200, 200))
            
            # Predict
            label, confidence = self.recognizer.predict(face_resized)
            
            # Classify as known or unknown
            if confidence < self.confidence_threshold:
                name = self.label_map.get(label, 'UNKNOWN')
                is_known = True
            else:
                name = 'UNKNOWN'
                is_known = False
                label = -1
            
            results.append({
                'name': name,
                'is_known': is_known,
                'confidence': round(float(confidence), 2),
                'bbox': (x, y, w, h),
                'label': label
            })
        
        return results
    
    def draw_results(self, frame, face_results):
        """
        Draw recognition results on frame.
        
        Args:
            frame: BGR image
            face_results: output from recognize_frame()
            
        Returns:
            annotated frame
        """
        annotated = frame.copy()
        
        for result in face_results:
            x, y, w, h = result['bbox']
            name = result['name']
            confidence = result['confidence']
            is_known = result['is_known']
            
            # Color
            if is_known:
                color = (0, 255, 0)      # Green = known
                label_text = f"{name} ({confidence:.0f})"
            else:
                color = (0, 0, 255)      # Red = unknown/stranger
                label_text = f"UNKNOWN ({confidence:.0f})"
            
            # Rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Label background
            text_size = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )[0]
            
            cv2.rectangle(
                annotated,
                (x, y - text_size[1] - 12),
                (x + text_size[0] + 4, y),
                color, -1
            )
            
            cv2.putText(
                annotated, label_text,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
        
        return annotated
    
    # ─────────────────────────────────────────────
    # STANDALONE DEMO
    # ─────────────────────────────────────────────
    
    def run_demo(self):
        """
        Standalone webcam demo.
        Run this to test recognition before pipeline integration.
        """
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("\nFace Recognition Demo")
        print(f"Known persons: {list(self.label_map.values())}")
        print(f"Threshold: {self.confidence_threshold} "
              f"(lower confidence = better match)")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run recognition every frame
            # (in pipeline we'll throttle this)
            results = self.recognize_frame(frame)
            
            # Draw
            annotated = self.draw_results(frame, results)
            
            # Info overlay
            known_count = sum(1 for r in results if r['is_known'])
            unknown_count = sum(1 for r in results if not r['is_known'])
            
            cv2.putText(annotated,
                       f"Known: {known_count} | Unknown: {unknown_count}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            
            cv2.putText(annotated,
                       f"Persons in DB: {list(self.label_map.values())}",
                       (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
            
            # Stranger warning banner
            if unknown_count > 0:
                h = annotated.shape[0]
                cv2.rectangle(annotated, (0, h-50), 
                             (annotated.shape[1], h), (0, 0, 200), -1)
                cv2.putText(annotated, 
                           "⚠ STRANGER DETECTED",
                           (10, h-15),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended.")


if __name__ == "__main__":
    fr = SimpleFaceRecognizer()
    
    # Force retrain with improved code
    success = fr.train(force_retrain=True)
    
    if success:
        fr.run_demo()
    else:

        print("\nFix dataset issues above, then run again.")

        