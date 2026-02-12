import cv2
from ultralytics import YOLO
import numpy as np
import time

# ===== YOLOv8 POSE MODEL LOAD =====
print("Loading YOLOv8 pose model...")
model = YOLO('yolov8n-pose.pt')
print("✅ Model loaded!")

# ===== VIDEO SOURCE =====
# Option 1: Webcam
# cap = cv2.VideoCapture(0)

# Option 2: Sample video
cap = cv2.VideoCapture(0)  # 0 for webcam, ya video file path

if not cap.isOpened():
    print("❌ Camera nahi khul rahi! Sample video use karte hain...")
    cap = cv2.VideoCapture(0)  # Try webcam again
    if not cap.isOpened():
        print("❌ Webcam bhi nahi khul raha! Fake mode on...")
        # Fake frame banao
        class FakeCap:
            def read(self):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            def release(self):
                pass
        cap = FakeCap()

print("🎥 Camera ready! Press 'q' to quit")

# ===== MAIN LOOP =====
fps_counter = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 pose detection
    results = model(frame, verbose=False)  # verbose=False to avoid printing
    
    # Draw poses on frame
    annotated_frame = results[0].plot()
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_time >= 1:
        fps = fps_counter
        fps_counter = 0
        fps_time = time.time()
    
    # Get keypoints count
    keypoints_count = 0
    if results[0].keypoints is not None:
        keypoints_count = len(results[0].keypoints.data)
    
    # Add text
    cv2.putText(annotated_frame, f"YOLOv8 Pose | FPS: {fps} | Athletes: {keypoints_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('SportIQ - YOLOv8 Real Pose Estimation', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
