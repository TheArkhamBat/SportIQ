import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

class SportsDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        
        if os.path.exists(os.path.join(self.data_dir, 'annotations.json')):
            with open(os.path.join(self.data_dir, 'annotations.json'), 'r') as f:
                annotations = json.load(f)
            
            for video_id, video_data in annotations.items():
                samples.append({
                    'video_path': os.path.join(self.data_dir, 'videos', f"{video_id}.mp4"),
                    'annotations': video_data
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames = self._load_video_frames(sample['video_path'])
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        annotations = sample['annotations']
        
        return {
            'frames': frames,
            'annotations': annotations,
            'video_id': os.path.basename(sample['video_path']).replace('.mp4', '')
        }
    
    def _load_video_frames(self, video_path, max_frames=100):
        if not os.path.exists(video_path):
            return [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(10)]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(10)]
        
        return frames

class SportsDataLoader:
    def __init__(self, config):
        self.config = config
    
    def create_data_loader(self, data_dir, batch_size=1, shuffle=True, num_workers=0):
        dataset = SportsDataset(data_dir)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def load_pose_data(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def save_analysis_results(self, results, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def export_to_csv(self, data, filepath):
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)