import cv2
from core.pose_estimator import PoseEstimator
from utils.visualization import VisualizationEngine

# Initialize
print("🎥 Starting SportIQ Real-Time...")
pose_estimator = PoseEstimator()
viz = VisualizationEngine()

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera ready! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera error")
        break
    
    # Detect pose
    pose = pose_estimator.estimate_pose(frame)
    
    # Draw visualization
    viz_frame = viz.draw_pose_on_frame(frame, pose)
    
    # Show FPS
    cv2.putText(viz_frame, "Press 'q' to quit", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display
    cv2.imshow('SportIQ - Real-Time Pose Estimation', viz_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Realtime stopped")
