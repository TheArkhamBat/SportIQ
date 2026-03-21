import numpy as np
from collections import defaultdict

class PerformanceTracker:
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.performance_baselines = {}
    
    def track_athlete_performance(self, athlete_id, session_data):
        session_metrics = self._calculate_session_metrics(session_data)
        
        for metric, value in session_metrics.items():
            self.metrics_history[f"{athlete_id}_{metric}"].append(value)
        
        return session_metrics
    
    def _calculate_session_metrics(self, session_data):
        poses = session_data.get('poses', [])
        velocities = session_data.get('velocities', [])
        accelerations = session_data.get('accelerations', [])
        angles = session_data.get('angles', [])
        
        metrics = {}
        
        metrics['total_frames'] = len(poses)
        metrics['detection_confidence'] = self._calculate_average_confidence(poses)
        metrics['movement_efficiency'] = self._calculate_movement_efficiency(poses, velocities)
        metrics['power_output'] = self._estimate_power_output(accelerations)
        metrics['flexibility_score'] = self._calculate_flexibility(angles)
        metrics['balance_metric'] = self._calculate_balance(poses)
        metrics['fatigue_index'] = self._detect_fatigue(velocities)
        
        return metrics
    
    def _calculate_average_confidence(self, poses):
        if not poses:
            return 0
        
        total_confidence = 0
        total_joints = 0
        
        for pose in poses:
            for joint_data in pose.values():
                total_confidence += joint_data['confidence']
                total_joints += 1
        
        return total_confidence / total_joints if total_joints > 0 else 0
    
    def _calculate_movement_efficiency(self, poses, velocities):
        if len(velocities) < 2:
            return 0
        
        total_efficiency = 0
        count = 0
        
        for i in range(1, len(poses)):
            upper_body_velocity = self._get_upper_body_velocity(velocities[i-1])
            lower_body_velocity = self._get_lower_body_velocity(velocities[i-1])
            
            if upper_body_velocity > 0 and lower_body_velocity > 0:
                efficiency = lower_body_velocity / upper_body_velocity
                total_efficiency += min(efficiency, 2.0)
                count += 1
        
        return total_efficiency / count if count > 0 else 0
    
    def _get_upper_body_velocity(self, velocity_frame):
        upper_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        velocities = [velocity_frame[j] for j in upper_joints if j in velocity_frame]
        return np.mean(velocities) if velocities else 0
    
    def _get_lower_body_velocity(self, velocity_frame):
        lower_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        velocities = [velocity_frame[j] for j in lower_joints if j in velocity_frame]
        return np.mean(velocities) if velocities else 0
    
    def _estimate_power_output(self, accelerations):
        if not accelerations:
            return 0
        
        total_power = 0
        count = 0
        
        for acc_frame in accelerations:
            if acc_frame:
                avg_acceleration = np.mean(list(acc_frame.values()))
                power = avg_acceleration * 75
                total_power += power
                count += 1
        
        return total_power / count if count > 0 else 0
    
    def _calculate_flexibility(self, angles):
        if not angles:
            return 0
        
        flexibility_scores = []
        
        for angle_frame in angles:
            for angle_name, angle_value in angle_frame.items():
                if 'knee' in angle_name or 'elbow' in angle_name:
                    flexibility = min(angle_value / 180, 1.0)
                    flexibility_scores.append(flexibility)
        
        return np.mean(flexibility_scores) if flexibility_scores else 0
    
    def _calculate_balance(self, poses):
        if not poses:
            return 0
        
        balance_scores = []
        
        for pose in poses:
            if pose['left_ankle']['confidence'] > 0.5 and pose['right_ankle']['confidence'] > 0.5:
                left_y = pose['left_ankle']['y']
                right_y = pose['right_ankle']['y']
                balance = 1 - abs(left_y - right_y)
                balance_scores.append(max(balance, 0))
        
        return np.mean(balance_scores) if balance_scores else 0
    
    def _detect_fatigue(self, velocities):
        if len(velocities) < 10:
            return 0
        
        first_half = velocities[:len(velocities)//2]
        second_half = velocities[len(velocities)//2:]
        
        first_avg = self._calculate_average_velocity(first_half)
        second_avg = self._calculate_average_velocity(second_half)
        
        if first_avg > 0:
            return (first_avg - second_avg) / first_avg
        return 0
    
    def _calculate_average_velocity(self, velocities):
        if not velocities:
            return 0
        
        total_velocity = 0
        count = 0
        
        for frame in velocities:
            if frame:
                frame_avg = np.mean(list(frame.values()))
                total_velocity += frame_avg
                count += 1
        
        return total_velocity / count if count > 0 else 0
    
    def get_performance_trends(self, athlete_id, metric, window=5):
        key = f"{athlete_id}_{metric}"
        if key not in self.metrics_history or len(self.metrics_history[key]) < window:
            return None
        
        values = self.metrics_history[key][-window:]
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        return {
            'current': values[-1],
            'trend': trend,
            'improvement': trend > 0
        }
    
    def generate_performance_report(self, athlete_id):
        report = {
            'athlete_id': athlete_id,
            'metrics': {},
            'recommendations': []
        }
        
        for key in self.metrics_history:
            if key.startswith(athlete_id):
                metric_name = key.replace(f"{athlete_id}_", "")
                trends = self.get_performance_trends(athlete_id, metric_name)
                report['metrics'][metric_name] = trends
        
        self._generate_recommendations(report)
        return report
    
    def _generate_recommendations(self, report):
        metrics = report['metrics']
        
        if metrics.get('fatigue_index', {}).get('current', 0) > 0.3:
            report['recommendations'].append("High fatigue detected - consider rest and recovery")
        
        if metrics.get('movement_efficiency', {}).get('current', 0) < 0.7:
            report['recommendations'].append("Focus on movement efficiency and technique")
        
        if metrics.get('flexibility_score', {}).get('current', 0) < 0.6:
            report['recommendations'].append("Incorporate flexibility training")
        
        if metrics.get('balance_metric', {}).get('current', 0) < 0.8:
            report['recommendations'].append("Balance training recommended")