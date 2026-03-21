import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path=None):
        print("Loading YOLOv8 pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections (which joints to connect with lines)
        self.skeleton = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        print("Model ready!")
    
    def estimate_pose(self, image):
        results = self.model(image, verbose=False)
        pose_dict = {}
        
        if len(results) > 0 and results[0].keypoints is not None:
            if len(results[0].keypoints.data) > 0:
                keypoints = results[0].keypoints.data[0].cpu().numpy()
                for i, name in enumerate(self.keypoint_names):
                    if i < len(keypoints):
                        x, y, conf = keypoints[i]
                        pose_dict[name] = {'x': float(x), 'y': float(y), 'confidence': float(conf)}
                    else:
                        pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
            else:
                for name in self.keypoint_names:
                    pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
        else:
            for name in self.keypoint_names:
                pose_dict[name] = {'x': 0, 'y': 0, 'confidence': 0}
        
        return pose_dict
    
    def draw_pose(self, frame, pose):
        """Draw skeleton and keypoints on frame"""
        h, w = frame.shape[:2]
        
        # Draw connections (skeleton)
        for j1, j2 in self.skeleton:
            if j1 in pose and j2 in pose:
                p1 = pose[j1]
                p2 = pose[j2]
                if p1['confidence'] > 0.5 and p2['confidence'] > 0.5:
                    x1, y1 = int(p1['x']), int(p1['y'])
                    x2, y2 = int(p2['x']), int(p2['y'])
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw keypoints (joints)
        for joint, data in pose.items():
            if data['confidence'] > 0.5:
                x, y = int(data['x']), int(data['y'])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        return frame