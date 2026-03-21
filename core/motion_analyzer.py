import numpy as np
from scipy import signal
from scipy.spatial.distance import cosine

class MotionAnalyzer:
    def __init__(self):
        # Define limbs for velocity tracking
        self.limbs = {
            'right_arm': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_arm': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_leg': ('right_hip', 'right_knee', 'right_ankle'),
            'left_leg': ('left_hip', 'left_knee', 'left_ankle'),
            'torso': ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')
        }
        self.limb_velocities = {'right_arm': [], 'left_arm': [], 'right_leg': [], 'left_leg': []}
    
    def calculate_limb_velocity(self, pose1, pose2, fps=30):
        """Calculate velocity for each limb"""
        velocities = {}
        dt = 1.0 / fps
        
        for limb_name, joints in self.limbs.items():
            if limb_name in ['right_arm', 'left_arm', 'right_leg', 'left_leg']:
                # Use end point (wrist/ankle) for velocity
                end_joint = joints[-1]
                if end_joint in pose1 and end_joint in pose2:
                    if pose1[end_joint]['confidence'] > 0.5 and pose2[end_joint]['confidence'] > 0.5:
                        dx = pose2[end_joint]['x'] - pose1[end_joint]['x']
                        dy = pose2[end_joint]['y'] - pose1[end_joint]['y']
                        velocity = np.sqrt(dx**2 + dy**2) * fps
                        velocities[limb_name] = velocity
                        
                        # Store for averaging
                        self.limb_velocities[limb_name].append(velocity)
                        if len(self.limb_velocities[limb_name]) > 30:
                            self.limb_velocities[limb_name].pop(0)
        
        return velocities
    
    def get_average_velocities(self):
        """Get average velocities for each limb"""
        avg = {}
        for limb, vlist in self.limb_velocities.items():
            if vlist:
                avg[limb] = sum(vlist) / len(vlist)
            else:
                avg[limb] = 0
        return avg
    
    def calculate_joint_angles(self, pose):
        angles = {}
        
        # Right arm angle
        if all(j in pose for j in ['right_shoulder', 'right_elbow', 'right_wrist']):
            a = np.array([pose['right_shoulder']['x'], pose['right_shoulder']['y']])
            b = np.array([pose['right_elbow']['x'], pose['right_elbow']['y']])
            c = np.array([pose['right_wrist']['x'], pose['right_wrist']['y']])
            angles['right_elbow'] = self._angle_between(a, b, c)
        
        # Left arm angle
        if all(j in pose for j in ['left_shoulder', 'left_elbow', 'left_wrist']):
            a = np.array([pose['left_shoulder']['x'], pose['left_shoulder']['y']])
            b = np.array([pose['left_elbow']['x'], pose['left_elbow']['y']])
            c = np.array([pose['left_wrist']['x'], pose['left_wrist']['y']])
            angles['left_elbow'] = self._angle_between(a, b, c)
        
        # Right knee angle
        if all(j in pose for j in ['right_hip', 'right_knee', 'right_ankle']):
            a = np.array([pose['right_hip']['x'], pose['right_hip']['y']])
            b = np.array([pose['right_knee']['x'], pose['right_knee']['y']])
            c = np.array([pose['right_ankle']['x'], pose['right_ankle']['y']])
            angles['right_knee'] = self._angle_between(a, b, c)
        
        # Left knee angle
        if all(j in pose for j in ['left_hip', 'left_knee', 'left_ankle']):
            a = np.array([pose['left_hip']['x'], pose['left_hip']['y']])
            b = np.array([pose['left_knee']['x'], pose['left_knee']['y']])
            c = np.array([pose['left_ankle']['x'], pose['left_ankle']['y']])
            angles['left_knee'] = self._angle_between(a, b, c)
        
        return angles
    
    def _angle_between(self, a, b, c):
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def assess_injury_risk(self, angles, velocities):
        """Generate injury risk assessment based on angles and velocities"""
        risks = []
        overall = "LOW"
        
        # Knee angle risk (ACL risk)
        for side in ['right', 'left']:
            knee = angles.get(f'{side}_knee', 90)
            if knee < 45:
                risks.append(f"{side}_knee: HIGH - ACL risk ({knee:.1f}°)")
                overall = "HIGH"
            elif knee < 60:
                risks.append(f"{side}_knee: MODERATE - Patellar stress ({knee:.1f}°)")
                if overall != "HIGH":
                    overall = "MODERATE"
        
        # Elbow hyperextension risk
        for side in ['right', 'left']:
            elbow = angles.get(f'{side}_elbow', 160)
            if elbow > 170:
                risks.append(f"{side}_elbow: MODERATE - Hyperextension ({elbow:.1f}°)")
                if overall != "HIGH":
                    overall = "MODERATE"
            elif elbow < 140:
                risks.append(f"{side}_elbow: LOW - Limited range ({elbow:.1f}°)")
        
        # Velocity-based fatigue detection
        if velocities:
            fast_limbs = [k for k, v in velocities.items() if v > 50]
            if len(fast_limbs) > 2:
                risks.append(f"High velocity in {', '.join(fast_limbs)} - fatigue risk")
        
        return {'overall': overall, 'findings': risks[:3]}
    
    def analyze_gait(self, pose_sequence, fps=30):
        """Basic gait analysis from ankle movement"""
        if len(pose_sequence) < 10:
            return {'step_symmetry': 0, 'cadence': 0}
        
        left_ankle_y = []
        right_ankle_y = []
        
        for pose in pose_sequence:
            if 'left_ankle' in pose and pose['left_ankle']['confidence'] > 0.5:
                left_ankle_y.append(pose['left_ankle']['y'])
            if 'right_ankle' in pose and pose['right_ankle']['confidence'] > 0.5:
                right_ankle_y.append(pose['right_ankle']['y'])
        
        if len(left_ankle_y) < 10 or len(right_ankle_y) < 10:
            return {'step_symmetry': 0, 'cadence': 0}
        
        # Calculate symmetry
        min_len = min(len(left_ankle_y), len(right_ankle_y))
        left = np.array(left_ankle_y[:min_len])
        right = np.array(right_ankle_y[:min_len])
        left = (left - np.mean(left)) / (np.std(left) + 1e-6)
        right = (right - np.mean(right)) / (np.std(right) + 1e-6)
        
        try:
            symmetry = 1 - cosine(left, right)
        except:
            symmetry = 0.5
        
        # Simple cadence estimation
        if len(left_ankle_y) > 15:
            peaks, _ = signal.find_peaks(left_ankle_y, distance=10)
            if len(peaks) > 1:
                avg_steps = len(peaks) / (len(pose_sequence) / fps)
                cadence = avg_steps * 60
            else:
                cadence = 0
        else:
            cadence = 0
        
        return {'step_symmetry': max(0, min(1, symmetry)), 'cadence': round(cadence, 1)}