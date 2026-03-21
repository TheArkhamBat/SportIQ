import cv2
import numpy as np
import torch
from typing import List, Dict, Any

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_frames(self, video_path: str, target_fps: int = 30) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(original_fps / target_fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def preprocess_frame(self, frame: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    
    def detect_players(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        height, width = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        detections = []
        
        dummy_detection = {
            'bbox': [width//4, height//4, width//2, height//2],
            'confidence': 0.9,
            'class_id': 0
        }
        detections.append(dummy_detection)
        
        return detections
    
    def track_players(self, frames: List[np.ndarray]) -> Dict[int, List[Dict]]:
        tracks = {}
        next_track_id = 0
        
        for i, frame in enumerate(frames):
            detections = self.detect_players(frame)
            
            for det in detections:
                if next_track_id not in tracks:
                    tracks[next_track_id] = []
                
                tracks[next_track_id].append({
                    'frame_idx': i,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
                
                next_track_id += 1
        
        return tracks
    
    def create_player_crops(self, frames: List[np.ndarray], tracks: Dict) -> Dict[int, List[np.ndarray]]:
        player_crops = {}
        
        for track_id, track_data in tracks.items():
            crops = []
            for detection in track_data:
                frame_idx = detection['frame_idx']
                bbox = detection['bbox']
                
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                if (0 <= x < frames[0].shape[1] and 0 <= y < frames[0].shape[0] and
                    w > 0 and h > 0 and x + w <= frames[0].shape[1] and y + h <= frames[0].shape[0]):
                    
                    crop = frames[frame_idx][y:y+h, x:x+w]
                    if crop.size > 0:
                        crops.append(crop)
            
            player_crops[track_id] = crops
        
        return player_crops
    
    def optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return magnitude