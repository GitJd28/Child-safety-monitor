# scripts/evaluate_system.py
"""
Measures system accuracy for presentation.
Tests face recognition on held-out images.
"""

import cv2
import os
import sys
import numpy as np

sys.path.insert(0, 'src')
from detection.opencv_face_recognizer import SimpleFaceRecognizer

def evaluate_face_recognition():
    """
    Test recognition accuracy on your dataset.
    Uses 80% for training, 20% for testing.
    """
    
    print("="*50)
    print("FACE RECOGNITION ACCURACY EVALUATION")
    print("="*50)
    
    dataset_path = "data/face_dataset"
    recognizer = SimpleFaceRecognizer()
    
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 
        "haarcascade_frontalface_default.xml"
    )
    
    results = {
        'true_positive': 0,   # Known correctly identified
        'false_positive': 0,  # Unknown called known
        'true_negative': 0,   # Unknown correctly rejected
        'false_negative': 0,  # Known called unknown
        'total_tested': 0
    }
    
    per_person = {}
    
    persons = [
        p for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p))
        and not p.startswith('_')
        and not p.endswith('_backup')
    ]
    
    print(f"\nTesting on: {persons}")
    print("Using last 20% of each person's photos as test set\n")
    
    for person_name in persons:
        person_path = os.path.join(dataset_path, person_name)
        images = sorted([
            f for f in os.listdir(person_path) 
            if f.endswith('.jpg')
        ])
        
        if len(images) < 5:
            print(f"⚠️  {person_name}: too few images, skipping")
            continue
        
        # Use last 20% as test set
        test_start = int(len(images) * 0.8)
        test_images = images[test_start:]
        
        correct = 0
        total = 0
        
        for img_name in test_images:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Run recognition
            face_results = recognizer.recognize_frame(img)
            
            if not face_results:
                # No face detected in image
                results['false_negative'] += 1
                total += 1
                continue
            
            # Take best result
            best = face_results[0]
            predicted = best['name']
            is_known = best['is_known']
            
            total += 1
            results['total_tested'] += 1
            
            if is_known and predicted == person_name:
                correct += 1
                results['true_positive'] += 1
            elif is_known and predicted != person_name:
                results['false_positive'] += 1
            elif not is_known:
                results['false_negative'] += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        per_person[person_name] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
        
        print(f"  {person_name}: {correct}/{total} = {accuracy:.1f}%")
    
    # Overall metrics
    tp = results['true_positive']
    fp = results['false_positive']
    fn = results['false_negative']
    total = results['total_tested']
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall) 
          if (precision + recall) > 0 else 0)
    overall = tp / total * 100 if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS")
    print(f"{'='*50}")
    print(f"  Total tested:  {total} images")
    print(f"  Overall Acc:   {overall:.1f}%")
    print(f"  Precision:     {precision:.1f}%")
    print(f"  Recall:        {recall:.1f}%")
    print(f"  F1 Score:      {f1:.1f}%")
    print(f"{'='*50}")
    print()
    print("USE THESE NUMBERS IN YOUR PRESENTATION SLIDES")
    print(f"{'='*50}")
    
    return results, per_person


def evaluate_pose_detection():
    """
    Test pose detection manually.
    You perform poses, system labels them, you verify.
    """
    
    print("\n" + "="*50)
    print("POSE DETECTION MANUAL EVALUATION")
    print("="*50)
    
    sys.path.insert(0, 'src')
    from detection.pose_analyzer import PoseAnalyzer
    
    analyzer = PoseAnalyzer()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    test_poses = [
        ('standing', 'Stand normally upright'),
        ('reaching_high', 'Raise both hands above head'),
        ('fallen', 'Lie down or crouch very low'),
    ]
    
    results = {}
    
    for pose_name, instruction in test_poses:
        print(f"\nTest pose: {pose_name.upper()}")
        print(f"Action: {instruction}")
        print("System will detect for 5 seconds...")
        input("Press ENTER when ready...")
        
        detections = []
        start = cv2.getTickCount()
        
        while True:
            elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            if elapsed > 5:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            persons = analyzer.analyze_frame(frame)
            
            for p in persons:
                detections.append(p['pose_type'])
            
            annotated = analyzer.draw_pose_analysis(frame, persons)
            remaining = max(0, 5 - int(elapsed))
            
            cv2.putText(annotated,
                       f"Perform: {instruction}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            
            cv2.putText(annotated,
                       f"Time remaining: {remaining}s",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow("Pose Evaluation", annotated)
            cv2.waitKey(1)
        
        # Calculate accuracy for this pose
        if detections:
            correct = sum(1 for d in detections if d == pose_name)
            accuracy = correct / len(detections) * 100
        else:
            accuracy = 0
        
        results[pose_name] = {
            'detections': len(detections),
            'correct': sum(1 for d in detections if d == pose_name),
            'accuracy': accuracy,
            'most_common': max(set(detections), 
                              key=detections.count) if detections else 'none'
        }
        
        print(f"  Result: {accuracy:.0f}% correct")
        print(f"  Most detected as: {results[pose_name]['most_common']}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*50}")
    print("POSE DETECTION RESULTS")
    print(f"{'='*50}")
    for pose, data in results.items():
        print(f"  {pose:20}: {data['accuracy']:.0f}%")
    print(f"{'='*50}")
    
    return results


if __name__ == "__main__":
    
    print("Child Safety Monitor - System Evaluation")
    print("="*50)
    print("\n1. Evaluate face recognition accuracy")
    print("2. Evaluate pose detection accuracy")
    print("3. Run both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    face_results = None
    pose_results = None
    
    if choice in ['1', '3']:
        face_results, per_person = evaluate_face_recognition()
    
    if choice in ['2', '3']:
        pose_results = evaluate_pose_detection()
    
    print("\n✅ Evaluation complete!")
    print("Use these numbers in your presentation.")

    