import cv2
import numpy as np
import torch
import torch.nn as nn

class PoseEstimator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def _load_model(self, model_path):
        from models.pose_models import PoseNet
        model = PoseNet()
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def estimate_pose(self, image):
        if isinstance(image, np.ndarray):
            image = self._preprocess_image(image)
        
        with torch.no_grad():
            image_tensor = torch.FloatTensor(image).to(self.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            keypoints = self.model(image_tensor)
            keypoints = keypoints.cpu().numpy()
            
        return self._format_keypoints(keypoints[0])
    
    def _preprocess_image(self, image):
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def _format_keypoints(self, keypoints):
        formatted = {}
        for i, name in enumerate(self.keypoint_names):
            x = keypoints[i * 2]
            y = keypoints[i * 2 + 1]
            confidence = keypoints[34 + i] if len(keypoints) > 34 else 1.0
            formatted[name] = {'x': x, 'y': y, 'confidence': confidence}
        return formatted
    
    def process_video(self, video_path, callback=None):
        cap = cv2.VideoCapture(video_path)
        poses = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            pose = self.estimate_pose(frame)
            poses.append(pose)
            
            if callback:
                callback(frame, pose)
                
        cap.release()
        return poses