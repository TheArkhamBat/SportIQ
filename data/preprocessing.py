import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
    
    def preprocess_poses(self, poses_sequence):
        if not poses_sequence:
            return np.array([])
        
        features = []
        
        for pose in poses_sequence:
            pose_features = self._extract_pose_features(pose)
            features.append(pose_features)
        
        features_array = np.array(features)
        
        if 'pose_features' not in self.scalers:
            self.scalers['pose_features'] = StandardScaler()
        
        if len(features_array) > 0:
            features_normalized = self.scalers['pose_features'].fit_transform(features_array)
            return features_normalized
        else:
            return np.array([])
    
    def _extract_pose_features(self, pose):
        features = []
        
        keypoints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        for joint in keypoints:
            if joint in pose and pose[joint]['confidence'] > 0.3:
                features.extend([pose[joint]['x'], pose[joint]['y']])
            else:
                features.extend([0.0, 0.0])
        
        if len(features) < 24:
            features.extend([0.0] * (24 - len(features)))
        
        return features[:24]
    
    def create_sequences(self, data, sequence_length=16):
        sequences = []
        
        for i in range(len(data) - sequence_length + 1):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def normalize_trajectories(self, trajectories, field_dimensions=(1.0, 1.0)):
        normalized_trajectories = {}
        
        for player_id, trajectory in trajectories.items():
            normalized_trajectory = []
            for point in trajectory:
                x_norm = point[0] / field_dimensions[0]
                y_norm = point[1] / field_dimensions[1]
                normalized_trajectory.append((x_norm, y_norm))
            normalized_trajectories[player_id] = normalized_trajectory
        
        return normalized_trajectories
    
    def augment_data(self, frames, poses):
        augmented_frames = []
        augmented_poses = []
        
        for frame, pose in zip(frames, poses):
            augmented_frames.append(frame)
            augmented_poses.append(pose)
            
            flip_frame = cv2.flip(frame, 1)
            flip_pose = self._flip_pose(pose)
            
            augmented_frames.append(flip_frame)
            augmented_poses.append(flip_pose)
        
        return augmented_frames, augmented_poses
    
    def _flip_pose(self, pose):
        flip_mapping = {
            'left_eye': 'right_eye',
            'right_eye': 'left_eye',
            'left_ear': 'right_ear', 
            'right_ear': 'left_ear',
            'left_shoulder': 'right_shoulder',
            'right_shoulder': 'left_shoulder',
            'left_elbow': 'right_elbow',
            'right_elbow': 'left_elbow',
            'left_wrist': 'right_wrist',
            'right_wrist': 'left_wrist',
            'left_hip': 'right_hip',
            'right_hip': 'left_hip',
            'left_knee': 'right_knee',
            'right_knee': 'left_knee',
            'left_ankle': 'right_ankle',
            'right_ankle': 'left_ankle'
        }
        
        flipped_pose = {}
        for joint, data in pose.items():
            if joint in flip_mapping:
                flipped_joint = flip_mapping[joint]
                flipped_pose[flipped_joint] = {
                    'x': 1.0 - data['x'],
                    'y': data['y'],
                    'confidence': data['confidence']
                }
            else:
                flipped_pose[joint] = {
                    'x': 1.0 - data['x'],
                    'y': data['y'],
                    'confidence': data['confidence']
                }
        
        return flipped_pose