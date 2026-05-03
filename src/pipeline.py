# src/pipeline.py
"""
Unified pipeline connecting all modules:
  - Person detection + child/adult classification
  - Pose estimation + dangerous pose detection  
  - Face recognition + known/unknown identification
  - Context engine + alert generation

This is the main file that runs the complete system.
"""

import cv2
import sys
import os
import time
import numpy as np
from collections import defaultdict
import csv
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.person_classifier import PersonClassifier
from src.detection.pose_analyzer import PoseAnalyzer
from src.detection.opencv_face_recognizer import SimpleFaceRecognizer


class ChildSafetyPipeline:
    
    def __init__(self):
        
        print("="*55)
        print(" CHILD SAFETY MONITOR - Initializing ")
        print("="*55)
        
        # Load all modules
        print("\n[1/3] Loading person detector...")
        self.person_classifier = PersonClassifier()
        print("  ✅ Person detector ready")
        
        print("\n[2/3] Loading pose analyzer...")
        self.pose_analyzer = PoseAnalyzer()
        print("  ✅ Pose analyzer ready")
        
        print("\n[3/3] Loading face recognizer...")
        self.face_recognizer = SimpleFaceRecognizer()
        success = self.face_recognizer.train()
        if not success:
            print("  ⚠️  Face recognizer not trained. Check dataset.")
        else:
            print("  ✅ Face recognizer ready")
        
        print("\n" + "="*55)
        print(" All modules loaded. Starting pipeline...")
        print("="*55 + "\n")
        
        # ── Frame Processing State ──
        self.frame_count = 0
        
        # Run face recognition every N frames
        # (it's slower than YOLO, so we throttle it)
        self.face_recognition_interval = 5
        
        # Store last face results (reuse between recognition frames)
        self.last_face_results = []
        
        # ── Context Tracking State ──
        # Track when child was first seen alone
        self.child_alone_since = None
        
        # Track when unknown person appeared
        self.unknown_appeared_at = None
        
        # Alert cooldowns (don't spam same alert)
        # {alert_type: last_triggered_time}
        self.alert_cooldowns = defaultdict(float)
        self.cooldown_seconds = 30
        
        # Alert log
        self.alert_log = []

        # ── Danger Zones ──
        self.danger_zones = {}
        self.defining_zone = False
    
    # ─────────────────────────────────────────────
    # CONTEXT ENGINE
    # ─────────────────────────────────────────────
    
    def analyze_context(self, persons, face_results, pose_results):
        """
        The brain of the system.
        
        Takes all detection results and decides:
        - What is happening in the scene?
        - Is there a dangerous situation?
        - Should an alert be triggered?
        
        Args:
            persons:      from person_classifier  → child/adult list
            face_results: from face_recognizer    → known/unknown list
            pose_results: from pose_analyzer      → pose type list
            
        Returns:
            alerts: list of alert dicts
            context: dict describing current scene
        """

        # DEBUG - remove after fixing
        print(f"DEBUG → persons:{len(persons)} "
              f"faces:{len(face_results)} "
              f"known:{sum(1 for f in face_results if f['is_known'])} "
              f"unknown:{sum(1 for f in face_results if not f['is_known'])}")
        
        current_time = time.time()
        alerts = []
        
        # ── Parse current scene ──
        num_children = sum(1 for p in persons if p['type'] == 'child')
        num_adults = sum(1 for p in persons if p['type'] == 'adult')
        
        num_known = sum(1 for f in face_results if f['is_known'])
        num_unknown = sum(1 for f in face_results if not f['is_known'])
        
        # Dangerous poses
        dangerous_poses = [
            p for p in pose_results 
            if p['danger_level'] >= 2
        ]
        fallen_poses = [
            p for p in pose_results 
            if p['pose_type'] == 'fallen'
        ]
        
        child_present = num_children > 0
        adult_present = num_adults > 0
        unknown_present = num_unknown > 0
        
        # ── Context summary ──
        context = {
            'num_children': num_children,
            'num_adults': num_adults,
            'num_known_faces': num_known,
            'num_unknown_faces': num_unknown,
            'child_present': child_present,
            'adult_present': adult_present,
            'unknown_present': unknown_present,
            'dangerous_poses': len(dangerous_poses),
            'fallen_detected': len(fallen_poses) > 0,
            'scene_safe': True,  # will update below
            'timestamp': current_time
        }
        
        # ════════════════════════════════════════
        # ALERT RULES (Priority Order)
        # ════════════════════════════════════════
        
        # ── RULE 1: Stranger with child (CRITICAL) ──
        # Unknown adult detected + child present
        # Only fire if we actually have face results
        if (unknown_present and
            child_present and
            len(face_results) > 0):    # ← KEY FIX
            alert = self._make_alert(
                alert_type='stranger_with_child',
                severity='CRITICAL',
                message='Unknown person detected near child!',
                action='Verify immediately who is with your child.',
                current_time=current_time
            )
            if alert:
                alerts.append(alert)
                context['scene_safe'] = False
        
        # ── RULE 2: Child fallen (HIGH) ──
        # Fallen pose detected + child present
        if fallen_poses and child_present:
            alert = self._make_alert(
                alert_type='child_fallen',
                severity='HIGH',
                message='Child may have fallen!',
                action='Check if child is okay.',
                current_time=current_time
            )
            if alert:
                alerts.append(alert)
                context['scene_safe'] = False
        
        # ── RULE 3: Child alone (MEDIUM) ──
        # Child present + no adult present
        if child_present and not adult_present:
            if self.child_alone_since is None:
                self.child_alone_since = current_time
            
            alone_duration = current_time - self.child_alone_since
            
            # Alert after 30 seconds alone
            if alone_duration >= 30:
                alert = self._make_alert(
                    alert_type='child_alone',
                    severity='MEDIUM',
                    message=f'Child has been alone for '
                            f'{int(alone_duration)}s',
                    action='Check on your child.',
                    current_time=current_time
                )
                if alert:
                    alerts.append(alert)
                    context['scene_safe'] = False
        else:
            # Reset alone timer when adult returns
            self.child_alone_since = None
        
        # ── RULE 4: Dangerous pose, child alone (HIGH) ──
        # Child doing dangerous thing + no adult watching
        if dangerous_poses and child_present and not adult_present:
            pose_types = [p['pose_type'] for p in dangerous_poses]
            alert = self._make_alert(
                alert_type='dangerous_pose_unsupervised',
                severity='HIGH',
                message=f'Child in dangerous pose '
                        f'({", ".join(pose_types)}) without supervision!',
                action='Check on child immediately.',
                current_time=current_time
            )
            if alert:
                alerts.append(alert)
                context['scene_safe'] = False
        
        # ── RULE 5: Unknown adult, no child (LOW) ──
        # Stranger present but no child in frame
        # Only fire if we actually have face results
        # AND someone is genuinely unknown
        # NOT just because face results are empty
        if (unknown_present and
            not child_present and
            len(face_results) > 0):    # ← KEY FIX
            alert = self._make_alert(
                alert_type='unknown_adult_no_child',
                severity='LOW',
                message='Unknown person in home (no child visible).',
                action='Verify who is at home.',
                current_time=current_time
            )
            if alert:
                alerts.append(alert)

        # ── Danger Zone Check ──
        zone_alerts = self.check_danger_zones(persons, current_time)
        alerts.extend(zone_alerts)
        
        return alerts, context
    
    
    def _make_alert(self, alert_type, severity, message, 
                    action, current_time, frame=None):
        """
        Create alert dict with cooldown check.
        Returns None if alert is in cooldown period.
        """
        last_time = self.alert_cooldowns[alert_type]
        
        if current_time - last_time < self.cooldown_seconds:
            return None  # Still in cooldown
        
        self.alert_cooldowns[alert_type] = current_time
        
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'action': action,
            'timestamp': current_time
        }
        
        self.alert_log.append(alert)
        self.save_alert_to_log(alert, frame)
        
        # Print to console
        icons = {
            'CRITICAL': '🚨',
            'HIGH':     '⚠️ ',
            'MEDIUM':   '🔔',
            'LOW':      'ℹ️ '
        }
        icon = icons.get(severity, '•')
        print(f"\n{icon} [{severity}] {message}")
        print(f"   → {action}")
        
        return alert

    # ─────────────────────────────────────────────
    # ALERT LOGGING
    # ─────────────────────────────────────────────

    def save_alert_to_log(self, alert, frame=None):
        """Save alert to CSV and auto-screenshot on critical alerts"""
        
        log_path = "outputs/alert_log.csv"
        os.makedirs("outputs", exist_ok=True)
        
        file_exists = os.path.exists(log_path)
        
        screenshot_path = ''
        
        # Auto screenshot for HIGH and CRITICAL
        if frame is not None and alert['severity'] in ['CRITICAL', 'HIGH']:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = f"outputs/alert_{alert['severity']}_{ts}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"  📸 Screenshot saved: {screenshot_path}")
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'severity',
                    'type', 'message', 'screenshot'
                ])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                alert['severity'],
                alert['type'],
                alert['message'],
                screenshot_path
            ])

    # ─────────────────────────────────────────────
    # DANGER ZONES
    # ─────────────────────────────────────────────

    def setup_danger_zones_interactive(self, frame):
        """
        Let user draw danger zones on first frame.
        Press 'd' during pipeline to define zones.
        
        Usage:
            Press 'd' → click and drag to draw zone
            Name the zone (kitchen/stairs/window)
            Zone saved, child entering = alert
        """
        
        zones = {}
        drawing = False
        start_point = None
        temp_frame = frame.copy()
        zone_name = input("\nEnter zone name (e.g. kitchen, stairs): ").strip()
        
        if not zone_name:
            return
        
        print(f"Draw '{zone_name}' zone: click and drag on the video window")
        print("Press ENTER when done, ESC to cancel")
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, temp_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = frame.copy()
                cv2.rectangle(temp, start_point, (x, y), (0, 0, 255), 2)
                cv2.putText(temp, zone_name, start_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow("Define Zone", temp)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                zones[zone_name] = (start_point, end_point)
                print(f"✅ Zone '{zone_name}' defined")
        
        cv2.namedWindow("Define Zone")
        cv2.setMouseCallback("Define Zone", mouse_callback)
        cv2.imshow("Define Zone", temp_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # Escape
                zones = {}
                break
        
        cv2.destroyWindow("Define Zone")
        self.danger_zones.update(zones)
        print(f"✅ Zones saved: {list(self.danger_zones.keys())}")


    def check_danger_zones(self, persons, current_time):
        """Check if any child is inside a defined danger zone"""
        
        alerts = []
        
        if not self.danger_zones:
            return alerts
        
        children = [p for p in persons if p['type'] == 'child']
        
        for child in children:
            x1, y1, x2, y2 = child['bbox']
            child_cx = (x1 + x2) // 2
            child_cy = (y1 + y2) // 2
            
            for zone_name, (top_left, bottom_right) in self.danger_zones.items():
                zx1, zy1 = top_left
                zx2, zy2 = bottom_right
                
                # Check if child center is inside zone
                if (zx1 <= child_cx <= zx2 and 
                    zy1 <= child_cy <= zy2):
                    
                    alert = self._make_alert(
                        alert_type=f'child_in_{zone_name}',
                        severity='HIGH',
                        message=f'Child entered danger zone: {zone_name}!',
                        action=f'Child is near {zone_name} unsupervised.',
                        current_time=current_time
                    )
                    if alert:
                        alerts.append(alert)
        
        return alerts
    
    # ─────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────
    
    def draw_pipeline_output(self, frame, persons, 
                              face_results, pose_results,
                              alerts, context):
        """
        Draw all detections and alerts on frame.
        Clean, organized display.
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # ── Draw danger zones ──
        for zone_name, (top_left, bottom_right) in self.danger_zones.items():
            cv2.rectangle(output, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(output, f"ZONE: {zone_name}",
                       (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)

        # ── Draw person bounding boxes ──
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            ptype = person['type']
            
            color = (0, 165, 255) if ptype == 'child' else (255, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            cv2.putText(output, ptype.upper(),
                       (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)
        
        # ── Draw face recognition results ──
        for face in face_results:
            x, y, fw, fh = face['bbox']
            is_known = face['is_known']
            name = face['name']
            conf = face['confidence']
            
            color = (0, 255, 0) if is_known else (0, 0, 255)
            
            cv2.rectangle(output, (x, y), (x+fw, y+fh), color, 3)
            
            label = f"{name} ({conf:.0f})"
            cv2.putText(output, label,
                       (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
        
        # ── Draw pose danger indicators ──
        for pose in pose_results:
            if pose['danger_level'] >= 2:
                x1, y1, x2, y2 = pose['bbox']
                cv2.rectangle(output, (x1, y1), (x2, y2), 
                             (0, 0, 255), 4)
                cv2.putText(output, 
                           f"POSE: {pose['pose_type'].upper()}",
                           (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 255), 2)
        
        # ── Info panel (top left) ──
        panel_lines = [
            f"People: {context['num_children']} child, "
            f"{context['num_adults']} adult",
            f"Faces:  {context['num_known_faces']} known, "
            f"{context['num_unknown_faces']} unknown",
        ]
        
        if self.child_alone_since:
            alone_sec = int(time.time() - self.child_alone_since)
            panel_lines.append(f"Child alone: {alone_sec}s")
        
        for i, line in enumerate(panel_lines):
            cv2.putText(output, line,
                       (10, 30 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.65, (255, 255, 255), 2)
        
        # ── Alert banner (bottom of frame) ──
        if alerts:
            # Get highest severity
            severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            severities = [a['severity'] for a in alerts]
            
            top_severity = next(
                (s for s in severity_order if s in severities), 
                'LOW'
            )
            
            banner_colors = {
                'CRITICAL': (0, 0, 200),
                'HIGH':     (0, 100, 200),
                'MEDIUM':   (0, 165, 255),
                'LOW':      (100, 100, 100)
            }
            
            banner_color = banner_colors.get(top_severity, (100,100,100))
            
            # Banner background
            banner_h = 60
            cv2.rectangle(output, (0, h - banner_h), 
                         (w, h), banner_color, -1)
            
            # Alert text
            top_alert = next(
                a for a in alerts if a['severity'] == top_severity
            )
            
            cv2.putText(output,
                       f"⚠ {top_alert['message']}",
                       (10, h - 35),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            
            cv2.putText(output,
                       top_alert['action'],
                       (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (220, 220, 220), 1)
        
        else:
            # Safe indicator
            cv2.rectangle(output, (0, h-35), (w, h), (0, 120, 0), -1)
            cv2.putText(output, "✓ Scene Safe",
                       (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
        
        return output
    
    # ─────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────
    
    def run(self):
        """
        Main pipeline loop.
        Opens webcam and runs all modules every frame.
        Press 'q' to quit, 's' to save screenshot.
        """
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(" !!Cannot open webcam !!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Pipeline running. Controls:")
        print("  'q' = quit")
        print("  's' = save screenshot")
        print("  'r' = retrain face recognizer")
        print("  'l' = show alert log")
        print("  'd' = define danger zone\n")
        
        fps_times = []
        
        while True:
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(" !!! Failed to read frame !!!")
                break
            
            self.frame_count += 1
            
            # ── MODULE 1: Person detection ──
            persons = self.person_classifier.detect_and_classify(frame)
            
            # ── MODULE 2: Pose analysis ──
            pose_results = self.pose_analyzer.analyze_frame(frame)
            
            # ── MODULE 3: Face recognition ──
            # Throttled: only every N frames (slower model)
            if self.frame_count % self.face_recognition_interval == 0:
                self.last_face_results = \
                    self.face_recognizer.recognize_frame(frame)
            
            face_results = self.last_face_results
            
            # ── MODULE 4: Context + Alerts ──
            alerts, context = self.analyze_context(
                persons, face_results, pose_results
            )
            
            # ── Draw everything ──
            output = self.draw_pipeline_output(
                frame, persons, face_results, 
                pose_results, alerts, context
            )
            
            # ── FPS display ──
            fps_times.append(time.time() - t_start)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            cv2.putText(output, f"FPS: {fps:.1f}",
                       (output.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (200, 200, 200), 1)
            
            cv2.imshow("Child Safety Monitor", output)
            
            # ── Key controls ──
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('s'):
                # Save screenshot
                os.makedirs("outputs", exist_ok=True)
                fname = f"outputs/screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, output)
                print(f"📸 Screenshot saved: {fname}")
            
            elif key == ord('r'):
                print("\nRetraining face recognizer...")
                self.face_recognizer.train(force_retrain=True)
                print(" Retrained\n")
            
            elif key == ord('l'):
                print(f"\n{'='*40}")
                print(f"ALERT LOG ({len(self.alert_log)} total)")
                print(f"{'='*40}")
                for a in self.alert_log[-10:]:  # Last 10
                    t = time.strftime('%H:%M:%S', 
                                     time.localtime(a['timestamp']))
                    print(f"[{t}] {a['severity']:8} {a['message']}")
                print()

            elif key == ord('d'):
                print("\nDefine danger zone...")
                self.setup_danger_zones_interactive(frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n Pipeline stopped.")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Total alerts: {len(self.alert_log)}")


if __name__ == "__main__":
    pipeline = ChildSafetyPipeline()
    pipeline.run()