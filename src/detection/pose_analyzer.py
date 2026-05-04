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
                    'keypoints': numpy array (17, 2),
                    'pose_type': 'standing','fallen','climbing', etc.,
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
        keypoints = results[0].keypoints.xy.cpu().numpy()
        
        for i in range(len(boxes)):
            bbox = boxes[i].xyxy[0].cpu().numpy()
            kpts = keypoints[i]  # (17, 2)
            
            # Analyze pose
            pose_type, danger_level = self._classify_pose(kpts, bbox)
            
            persons.append({
                'bbox': tuple(map(int, bbox)),
                'keypoints': kpts,
                'pose_type': pose_type,
                'danger_level': danger_level
            })
        
        return persons
    
    def _is_valid(self, keypoint):
        """Check if a keypoint was actually detected (not 0,0)"""
        return not np.all(keypoint == 0)

    def _classify_pose(self, keypoints, bbox):
        """
        Classify pose based on keypoint positions.

        Returns:
            (pose_type, danger_level)
        """
        if np.all(keypoints == 0):
            return 'unknown', 0
        
        try:
            # ── Extract joints ──
            nose           = keypoints[self.KEYPOINTS['nose']]
            left_shoulder  = keypoints[self.KEYPOINTS['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
            left_hip       = keypoints[self.KEYPOINTS['left_hip']]
            right_hip      = keypoints[self.KEYPOINTS['right_hip']]
            left_knee      = keypoints[self.KEYPOINTS['left_knee']]
            right_knee     = keypoints[self.KEYPOINTS['right_knee']]
            left_ankle     = keypoints[self.KEYPOINTS['left_ankle']]
            right_ankle    = keypoints[self.KEYPOINTS['right_ankle']]
            left_wrist     = keypoints[self.KEYPOINTS['left_wrist']]
            right_wrist    = keypoints[self.KEYPOINTS['right_wrist']]

            # ── Require shoulders and hips as minimum ──
            if not all(self._is_valid(kp) for kp in 
                      [left_shoulder, right_shoulder,
                       left_hip, right_hip]):
                return 'unknown', 0

            # ── Midpoints ──
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid      = (left_hip + right_hip) / 2

            # ════════════════════════════════════════
            # RULE 1: FALLEN
            # ════════════════════════════════════════
            # Method A: Bounding box aspect ratio
            # A standing person is TALL (height > width)
            # A fallen person is WIDE (width > height)
            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Fallen if bbox is wider than it is tall
            bbox_fallen = (bbox_w > bbox_h * 1.2)

            # Method B: Shoulder-hip vertical vs horizontal span
            # When fallen, hips move to the SIDE of shoulders
            # not BELOW them
            sh_horizontal = abs(hip_mid[0] - shoulder_mid[0])
            sh_vertical   = abs(hip_mid[1] - shoulder_mid[1])

            # Relaxed ratio from 1.5 → 0.8
            keypoint_fallen = (sh_horizontal > sh_vertical * 0.8)

            # Method C: Nose drops to hip level or below
            # When lying down, head is at same height as hips
            nose_fallen = False
            if self._is_valid(nose) and self._is_valid(hip_mid):
                # In image coords Y increases downward
                # nose_y ≈ hip_y means person is horizontal
                nose_hip_diff = abs(nose[1] - hip_mid[1])
                nose_fallen = (nose_hip_diff < 80)  # pixels

            # Fallen if ANY TWO methods agree
            fallen_votes = sum([bbox_fallen, keypoint_fallen, nose_fallen])
            if fallen_votes >= 2:
                return 'fallen', 3

            # ════════════════════════════════════════
            # RULE 2: REACHING HIGH / CLIMBING
            # ════════════════════════════════════════
            wrist_raised = False

            if self._is_valid(left_wrist) and self._is_valid(right_wrist):
                # Both wrists above shoulders
                both_above = (left_wrist[1]  < shoulder_mid[1] and
                              right_wrist[1] < shoulder_mid[1])

                # At least one wrist significantly above head
                one_high = (left_wrist[1]  < shoulder_mid[1] - 40 or
                            right_wrist[1] < shoulder_mid[1] - 40)

                wrist_raised = both_above or one_high

            elif self._is_valid(left_wrist):
                wrist_raised = left_wrist[1] < shoulder_mid[1] - 40

            elif self._is_valid(right_wrist):
                wrist_raised = right_wrist[1] < shoulder_mid[1] - 40

            if wrist_raised:
                return 'reaching_high', 2

            # ════════════════════════════════════════
            # RULE 3: CROUCHING
            # ════════════════════════════════════════
            # Knees rise above hips in image (lower Y value)
            crouching = False

            if (self._is_valid(left_knee) and
                self._is_valid(right_knee)):

                knee_mid_y = (left_knee[1] + right_knee[1]) / 2

                # Knee Y < hip Y means knees are higher in frame
                # which happens when crouching or sitting
                if knee_mid_y < hip_mid[1] - 20:
                    crouching = True

            # Also check using bounding box compression
            # Crouching person has compressed height
            if bbox_h > 0 and bbox_w > 0:
                if bbox_h < bbox_w * 1.1 and not crouching:
                    crouching = True

            if crouching:
                return 'crouching', 1

            # ════════════════════════════════════════
            # RULE 4: STANDING / SITTING (safe)
            # ════════════════════════════════════════
            return 'standing', 0

        except Exception as e:
            return 'unknown', 0
    
    def draw_pose_analysis(self, frame, persons):
        """Draw pose annotations"""
        annotated = frame.copy()
        
        # Danger colors
        danger_colors = {
            0: (0, 255, 0),    # Green  - safe
            1: (0, 255, 255),  # Yellow - low
            2: (0, 165, 255),  # Orange - medium
            3: (0, 0, 255)     # Red    - high
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
    print("  - Stand normally       → 'standing'      (green)")
    print("  - Raise both hands     → 'reaching_high' (orange)")
    print("  - Crouch / sit down    → 'crouching'     (yellow)")
    print("  - Lie on floor/couch   → 'fallen'        (red)")
    print("  - Lean sideways        → 'fallen'        (red)")
    print("Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        persons = analyzer.analyze_frame(frame)
        annotated = analyzer.draw_pose_analysis(frame, persons)

        # Print pose info to console for debugging
        for p in persons:
            print(f"  Pose: {p['pose_type']:15} "
                  f"Danger: {p['danger_level']}  "
                  f"BBox: {p['bbox']}")
        
        # Draw YOLO skeleton on right panel
        results = analyzer.model(frame, verbose=False)
        skeleton_frame = results[0].plot()
        
        # Side by side
        combined = np.hstack([annotated, skeleton_frame])
        
        cv2.imshow('Pose Analysis (Left) | Skeleton (Right)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pose_analyzer()
    