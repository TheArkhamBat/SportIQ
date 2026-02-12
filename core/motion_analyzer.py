import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine

class MotionAnalyzer:
    def __init__(self):
        self.joint_connections = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), 
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
    
    def calculate_velocity(self, poses, fps=30):
        velocities = []
        for i in range(1, len(poses)):
            frame_velocity = {}
            for joint in poses[0].keys():
                if poses[i][joint]['confidence'] > 0.5 and poses[i-1][joint]['confidence'] > 0.5:
                    dx = poses[i][joint]['x'] - poses[i-1][joint]['x']
                    dy = poses[i][joint]['y'] - poses[i-1][joint]['y']
                    velocity = np.sqrt(dx**2 + dy**2) * fps
                    frame_velocity[joint] = velocity
            velocities.append(frame_velocity)
        return velocities
    
    def calculate_acceleration(self, velocities, fps=30):
        accelerations = []
        for i in range(1, len(velocities)):
            frame_acceleration = {}
            for joint in velocities[0].keys():
                if joint in velocities[i] and joint in velocities[i-1]:
                    dv = velocities[i][joint] - velocities[i-1][joint]
                    acceleration = dv * fps
                    frame_acceleration[joint] = acceleration
            accelerations.append(frame_acceleration)
        return accelerations
    
    def calculate_joint_angles(self, poses):
        angles = []
        for pose in poses:
            frame_angles = {}
            for conn in self.joint_connections:
                joint1, joint2, joint3 = self._get_angle_joints(conn)
                if all(j in pose and pose[j]['confidence'] > 0.5 for j in [joint1, joint2, joint3]):
                    angle = self._compute_angle(
                        pose[joint1], pose[joint2], pose[joint3]
                    )
                    frame_angles[f"{joint1}_{joint2}_{joint3}"] = angle
            angles.append(frame_angles)
        return angles
    
    def _get_angle_joints(self, connection):
        if connection == ('left_shoulder', 'left_elbow'):
            return 'left_shoulder', 'left_elbow', 'left_wrist'
        elif connection == ('right_shoulder', 'right_elbow'):
            return 'right_shoulder', 'right_elbow', 'right_wrist'
        elif connection == ('left_hip', 'left_knee'):
            return 'left_hip', 'left_knee', 'left_ankle'
        elif connection == ('right_hip', 'right_knee'):
            return 'right_hip', 'right_knee', 'right_ankle'
        else:
            return connection[0], connection[1], connection[1]
    
    def _compute_angle(self, point1, point2, point3):
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Handle zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def detect_impact(self, accelerations, threshold=50):
        impacts = []
        for i, acc_frame in enumerate(accelerations):
            if acc_frame:
                max_acc = max(acc_frame.values())
                if max_acc > threshold:
                    impacts.append({
                        'frame': i,
                        'max_acceleration': max_acc,
                        'joint': max(acc_frame, key=acc_frame.get)
                    })
        return impacts
    
    def analyze_gait(self, poses, fps=30):
        # Collect ankle positions
        left_ankle_y = []
        right_ankle_y = []
        
        for pose in poses:
            if 'left_ankle' in pose and pose['left_ankle']['confidence'] > 0.5:
                left_ankle_y.append(pose['left_ankle']['y'])
            if 'right_ankle' in pose and pose['right_ankle']['confidence'] > 0.5:
                right_ankle_y.append(pose['right_ankle']['y'])
        
        # If not enough data, return defaults
        if len(left_ankle_y) < 10 or len(right_ankle_y) < 10:
            return {
                'stride_frequency_left': 0,
                'stride_frequency_right': 0,
                'step_symmetry': 0.5
            }
        
        # Find peaks
        left_peaks, _ = signal.find_peaks(np.array(left_ankle_y), prominence=0.1)
        right_peaks, _ = signal.find_peaks(np.array(right_ankle_y), prominence=0.1)
        
        # Calculate stride times
        stride_times_left = []
        stride_times_right = []
        
        if len(left_peaks) > 1:
            stride_times_left = np.diff(left_peaks) / fps
        
        if len(right_peaks) > 1:
            stride_times_right = np.diff(right_peaks) / fps
        
        # Calculate frequencies
        stride_freq_left = 0
        stride_freq_right = 0
        
        if len(stride_times_left) > 0:
            mean_stride_left = np.mean(stride_times_left)
            if mean_stride_left > 0:
                stride_freq_left = 1.0 / mean_stride_left
        
        if len(stride_times_right) > 0:
            mean_stride_right = np.mean(stride_times_right)
            if mean_stride_right > 0:
                stride_freq_right = 1.0 / mean_stride_right
        
        # Calculate symmetry
        symmetry = self._calculate_symmetry(left_ankle_y, right_ankle_y)
        
        return {
            'stride_frequency_left': float(stride_freq_left),
            'stride_frequency_right': float(stride_freq_right),
            'step_symmetry': float(symmetry)
        }
    
    def _calculate_symmetry(self, signal1, signal2):
        if len(signal1) == 0 or len(signal2) == 0:
            return 0.5
        
        # Make signals same length
        min_len = min(len(signal1), len(signal2))
        sig1 = np.array(signal1[:min_len])
        sig2 = np.array(signal2[:min_len])
        
        # Normalize
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-6)
        sig2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-6)
        
        try:
            similarity = 1 - cosine(sig1, sig2)
            return float(np.clip(similarity, 0, 1))
        except:
            return 0.5
