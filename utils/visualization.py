import cv2
import numpy as np

class VisualizationEngine:
    def __init__(self):
        self.skeleton_connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
    
    def draw_pose_on_frame(self, frame, pose):
        """Draw skeleton and keypoints on frame"""
        # Make a copy
        viz_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw keypoints
        for joint_name, joint_data in pose.items():
            if joint_data['confidence'] > 0.5:
                x = int(joint_data['x'])
                y = int(joint_data['y'])
                
                # Ensure within frame
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                
                # Draw circle
                cv2.circle(viz_frame, (x, y), 4, (0, 255, 0), -1)
        
        # Draw skeleton
        for conn in self.skeleton_connections:
            if conn[0] in pose and conn[1] in pose:
                p1 = pose[conn[0]]
                p2 = pose[conn[1]]
                
                if p1['confidence'] > 0.5 and p2['confidence'] > 0.5:
                    x1 = int(p1['x'])
                    y1 = int(p1['y'])
                    x2 = int(p2['x'])
                    y2 = int(p2['y'])
                    
                    # Ensure within frame
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    cv2.line(viz_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Add title
        cv2.putText(viz_frame, "SportIQ - Real-Time Pose", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return viz_frame
    
    def create_performance_dashboard(self, performance_report):
        """Placeholder for dashboard"""
        print("📊 Dashboard generated")
        return None
