import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path=None):
        """YOLOv8 pose estimator for SportIQ main.py"""
        print("🚀 Loading YOLOv8 pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        
        # COCO keypoint names (17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        print("✅ YOLOv8 Pose Model Ready!")
    
    def estimate_pose(self, image):
        """Single frame pose estimation - returns dict format"""
        results = self.model(image, verbose=False)
        
        pose_dict = {}
        
        # CHECK - KOI PERSON DETECT HUA YA NAHI?
        if len(results) > 0 and results[0].keypoints is not None:
            # CHECK - KYA KEYPOINTS DATA HAI?
            if len(results[0].keypoints.data) > 0:
                # Get first person
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                
                for i, name in enumerate(self.keypoint_names):
                    if i < len(keypoints):
                        x, y, conf = keypoints[i]
                        pose_dict[name] = {
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf)
                        }
                    else:
                        pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
            else:
                # NO PERSON DETECTED - empty pose
                for name in self.keypoint_names:
                    pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
        else:
            # NO RESULTS - empty pose
            for name in self.keypoint_names:
                pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
        
        return pose_dict
    
    def process_video(self, video_path):
        """Process video and return list of poses"""
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose = self.estimate_pose(frame)
            poses.append(pose)
            frame_count += 1
            
            # Count frames with actual detections
            if pose[self.keypoint_names[0]]['confidence'] > 0:
                detection_count += 1
            
            if frame_count % 100 == 0:
                print(f"📹 Processed {frame_count} frames... (detections: {detection_count})")
        
        cap.release()
        print(f"✅ Total frames: {frame_count}, Frames with poses: {detection_count}")
        return poses
