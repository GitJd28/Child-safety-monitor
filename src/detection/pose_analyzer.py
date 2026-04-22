#4   3.2 
# src/detection/pose_analyzer.py
import cv2
from ultralytics import YOLO
import numpy as np

class PoseAnalyzer:
    """Analyze poses to detect dangerous activities"""
    
    # COCO keypoint indices
    KEYPOINTS = {
        'nose': 0,
        'left_eye': 1, 'right_eye': 2,
        'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')
    
    def analyze_frame(self, frame):
        """
        Detect persons and analyze their poses
        
        Returns:
            list of dicts: [
                {
                    'bbox': (x1, y1, x2, y2),
                    'keypoints': numpy array (17, 3),  # x, y, confidence
                    'pose_type': 'standing', 'fallen', 'climbing', etc.,
                    'danger_level': 0-3
                },
                ...
            ]
        """
        results = self.model(frame, verbose=False)
        
        persons = []
        
        if results[0].keypoints is None:
            return persons
        
        boxes = results[0].boxes
        keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: (num_people, 17, 2)
        
        for i in range(len(boxes)):
            bbox = boxes[i].xyxy[0].cpu().numpy()
            kpts = keypoints[i]  # (17, 2)
            
            # Analyze pose
            pose_type, danger_level = self._classify_pose(kpts)
            
            persons.append({
                'bbox': tuple(map(int, bbox)),
                'keypoints': kpts,
                'pose_type': pose_type,
                'danger_level': danger_level
            })
        
        return persons
    
    def _classify_pose(self, keypoints):
        """
        Classify pose based on keypoint positions
        
        Returns:
            (pose_type, danger_level)
        """
        # Check if keypoints are valid (not all zeros)
        if np.all(keypoints == 0):
            return 'unknown', 0
        
        # Extract key joints
        try:
            nose = keypoints[self.KEYPOINTS['nose']]
            left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
            left_hip = keypoints[self.KEYPOINTS['left_hip']]
            right_hip = keypoints[self.KEYPOINTS['right_hip']]
            left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
            right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
            left_ankle = keypoints[self.KEYPOINTS['left_ankle']]
            right_ankle = keypoints[self.KEYPOINTS['right_ankle']]
            
            # Skip if key joints not detected
            if any(np.all(kp == 0) for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return 'unknown', 0
            
            # Calculate midpoints
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2
            
            # 1. Check if person is horizontal (fallen)
            shoulder_hip_horizontal = abs(hip_mid[0] - shoulder_mid[0])
            shoulder_hip_vertical = abs(hip_mid[1] - shoulder_mid[1])
            
            if shoulder_hip_horizontal > shoulder_hip_vertical * 1.5:
                return 'fallen', 3  # High danger
            
            # 2. Check if hands raised high (climbing/reaching)
            wrist_avg_y = (left_wrist[1] + right_wrist[1]) / 2
            shoulder_avg_y = (left_shoulder[1] + right_shoulder[1]) / 2
            
            if wrist_avg_y > 0 and shoulder_avg_y > 0:  # Both detected
                if wrist_avg_y < shoulder_avg_y - 50:  # Hands significantly above shoulders
                    return 'reaching_high', 2  # Medium danger (could be climbing)
            
            # 3. Check if person is low to ground but not fallen
            if nose[1] > 0 and hip_mid[1] > 0:
                body_height = abs(nose[1] - hip_mid[1])
                if body_height < 100:  # Very compressed vertically
                    return 'crouching', 1  # Low danger
            
            # 4. Default: standing/sitting (safe)
            return 'standing', 0
            
        except Exception as e:
            return 'unknown', 0
    
    def draw_pose_analysis(self, frame, persons):
        """Draw pose annotations"""
        annotated = frame.copy()
        
        # Danger colors
        danger_colors = {
            0: (0, 255, 0),    # Green - safe
            1: (0, 255, 255),  # Yellow - low
            2: (0, 165, 255),  # Orange - medium
            3: (0, 0, 255)     # Red - high
        }
        
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            pose_type = person['pose_type']
            danger = person['danger_level']
            
            color = danger_colors.get(danger, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Label
            label = f"{pose_type.upper()} (Danger: {danger})"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated


def test_pose_analyzer():
    """Test pose analyzer with live webcam"""
    analyzer = PoseAnalyzer()
    
    cap = cv2.VideoCapture(0)
    print("Pose Analyzer Test")
    print("Try different poses:")
    print("  - Stand normally → 'standing' (green)")
    print("  - Raise hands high → 'reaching_high' (orange)")
    print("  - Lie down on floor → 'fallen' (red)")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze poses
        persons = analyzer.analyze_frame(frame)
        
        # Draw analysis
        annotated = analyzer.draw_pose_analysis(frame, persons)
        
        # Also draw skeleton (from YOLO)
        results = analyzer.model(frame, verbose=False)
        skeleton_frame = results[0].plot()
        
        # Combine both views side-by-side
        combined = np.hstack([annotated, skeleton_frame])
        
        cv2.imshow('Pose Analysis (Left) | Skeleton (Right)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pose_analyzer()
