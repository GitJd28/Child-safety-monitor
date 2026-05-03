# scripts/recapture_friend.py
"""
Recaptures friend photos using webcam.
Replaces the low-quality existing friend photos.
Run this when your friend is available.
If friend not available, captures YOUR face as a test 
for a second identity.
"""

import cv2
import os
import time
import shutil

def capture_face_dataset(person_name, num_photos=30):
    """
    Capture good quality face photos using webcam.
    Saves directly to data/face_dataset/{person_name}/
    
    Args:
        person_name: folder name (e.g., 'friend', 'parent2')
        num_photos: how many photos to take (30 recommended)
    """
    
    save_dir = f"data/face_dataset/{person_name}"
    
    # Backup existing photos if any
    if os.path.exists(save_dir):
        backup_dir = f"data/face_dataset/{person_name}_backup"
        if not os.path.exists(backup_dir):
            shutil.copytree(save_dir, backup_dir)
            print(f"✅ Backed up existing photos to {person_name}_backup/")
        
        # Clear existing
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
        print(f"✅ Cleared old photos from {save_dir}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Face detector
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return 0
    
    # Set good resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\n{'='*50}")
    print(f"CAPTURING: {person_name}")
    print(f"{'='*50}")
    print(f"Target: {num_photos} photos")
    print()
    print("IMPORTANT TIPS:")
    print("  ✅ Good lighting on face (not backlit)")
    print("  ✅ Move head slightly between each shot")
    print("  ✅ Try: straight, slight left, slight right")
    print("  ✅ Try: smile, neutral, slight different angles")
    print("  ✅ Stay 50-80cm from camera (not too close/far)")
    print()
    print("CONTROLS:")
    print("  SPACE = capture photo manually")
    print("  'a'   = toggle auto-capture (every 2 sec)")
    print("  'q'   = done / quit")
    print()
    
    input(f"Press ENTER when {person_name} is ready and in frame...")
    
    captured = 0
    auto_mode = False
    last_auto_time = time.time()
    face_detected_count = 0
    
    while captured < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for natural feeling
        display = cv2.flip(frame, 1)
        h, w = display.shape[:2]
        
        # Detect face to give feedback
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        face_found = len(faces) > 0
        
        # Draw face box if detected
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(display, (fx, fy), 
                         (fx+fw, fy+fh), (0, 255, 0), 2)
        
        # Face detection status
        if face_found:
            status_text = "✓ Face detected - ready to capture"
            status_color = (0, 255, 0)
        else:
            status_text = "✗ No face detected - adjust position"
            status_color = (0, 0, 255)
        
        # UI elements
        cv2.putText(display, f"Person: {person_name}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.putText(display, status_text,
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, status_color, 2)
        
        cv2.putText(display, f"Captured: {captured}/{num_photos}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 0), 2)
        
        mode_color = (0, 255, 0) if auto_mode else (180, 180, 180)
        cv2.putText(display, f"Auto: {'ON' if auto_mode else 'OFF'}",
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, mode_color, 2)
        
        cv2.putText(display, "SPACE=capture | a=auto | q=done",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
        
        # Progress bar
        bar_y = h - 40
        cv2.rectangle(display, (10, bar_y), (w-10, bar_y+15), (50,50,50), -1)
        progress_w = int((w - 20) * captured / num_photos)
        cv2.rectangle(display, (10, bar_y), (10+progress_w, bar_y+15), (0,255,0), -1)
        
        cv2.imshow(f"Capture: {person_name}", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle auto
        if key == ord('a'):
            auto_mode = not auto_mode
            print(f"  Auto mode: {'ON' if auto_mode else 'OFF'}")
        
        # Quit
        if key == ord('q'):
            print(f"\nStopped at {captured} photos")
            break
        
        # Decide whether to capture
        should_capture = False
        
        if key == ord(' ') and face_found:
            should_capture = True
        elif auto_mode and face_found and (time.time() - last_auto_time) >= 2.0:
            should_capture = True
            last_auto_time = time.time()
        elif key == ord(' ') and not face_found:
            print("  ⚠️  No face detected in frame - move closer or improve lighting")
        
        if should_capture:
            # Save original (not mirrored) frame
            filename = f"{person_name}_{captured:03d}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)  # non-mirrored
            
            captured += 1
            
            # Flash effect
            white = np.ones_like(display) * 255
            blend = cv2.addWeighted(display, 0.5, 
                                    white.astype(np.uint8), 0.5, 0)
            cv2.imshow(f"Capture: {person_name}", blend)
            cv2.waitKey(150)
            
            print(f"  📸 {captured}/{num_photos} captured")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Verify saved files
    saved = os.listdir(save_dir)
    avg_size = sum(
        os.path.getsize(os.path.join(save_dir, f)) 
        for f in saved
    ) / max(len(saved), 1) / 1024
    
    print(f"\n{'='*50}")
    print(f"✅ Capture complete for: {person_name}")
    print(f"   Photos saved: {len(saved)}")
    print(f"   Average file size: {avg_size:.1f} KB")
    
    if avg_size < 50:
        print(f"   ⚠️  WARNING: Average size {avg_size:.1f}KB is low.")
        print(f"   Try better lighting or move closer to camera.")
    else:
        print(f"   ✅ Good quality photos!")
    
    print(f"   Location: {os.path.abspath(save_dir)}")
    
    return len(saved)


if __name__ == "__main__":
    import sys
    
    print("Face Dataset Capture Tool")
    print("="*50)
    print("\nThis will capture new photos for recognition.")
    print("Existing photos will be backed up automatically.\n")
    
    print("Who do you want to capture?")
    print("  1. friend (recapture with better quality)")
    print("  2. New person (enter custom name)")
    print("  3. Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == '1':
        capture_face_dataset('friend', num_photos=30)
    
    elif choice == '2':
        name = input("Enter person name (no spaces, e.g. parent2): ").strip()
        if name:
            capture_face_dataset(name, num_photos=30)
        else:
            print("❌ Invalid name")
    
    elif choice == '3':
        capture_face_dataset('friend', num_photos=30)
        cont = input("\nCapture next person? (y/n): ").strip()
        if cont == 'y':
            name = input("Enter name: ").strip()
            capture_face_dataset(name, num_photos=30)
    
    else:
        print("Invalid choice")
    
    print("\n✅ Done! Next step:")
    print("   python src/detection/opencv_face_recognizer.py")