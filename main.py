import cv2
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer

class SportIQ:
    def __init__(self):
        self.pose = PoseEstimator()
        self.motion = MotionAnalyzer()
        self.prev_pose = None
        self.pose_history = []
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def draw_info_panel(self, frame, velocities, angles, injury, gait):
        """Draw analysis panel on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, "SPORTIQ - LIVE ANALYSIS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Velocities
        y += 30
        cv2.putText(frame, "VELOCITIES (px/s):", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 25
        for limb, vel in velocities.items():
            cv2.putText(frame, f"  {limb}: {vel:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        
        # Joint Angles
        y += 5
        cv2.putText(frame, "JOINT ANGLES (deg):", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 25
        for joint, ang in list(angles.items())[:4]:
            cv2.putText(frame, f"  {joint}: {ang:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
        
        # Injury Risk
        y += 5
        risk_color = (0, 255, 0) if injury['overall'] == 'LOW' else (0, 255, 255) if injury['overall'] == 'MODERATE' else (0, 0, 255)
        cv2.putText(frame, f"INJURY RISK: {injury['overall']}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        y += 25
        for finding in injury['findings'][:2]:
            cv2.putText(frame, f"  {finding[:35]}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 18
        
        # Gait
        y += 5
        cv2.putText(frame, f"GAIT SYMMETRY: {gait['step_symmetry']*100:.0f}%", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        y += 20
        cv2.putText(frame, f"CADENCE: {gait['cadence']} steps/min", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create fullscreen window
        cv2.namedWindow("SportIQ", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("SportIQ", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("SportIQ Live Analysis")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get pose
            pose = self.pose.estimate_pose(frame)
            self.pose_history.append(pose)
            if len(self.pose_history) > 30:
                self.pose_history.pop(0)
            
            # Draw skeleton
            frame = self.pose.draw_pose(frame, pose)
            
            # Analysis
            velocities = {}
            angles = {}
            injury = {'overall': 'LOW', 'findings': []}
            gait = {'step_symmetry': 0, 'cadence': 0}
            
            if self.prev_pose:
                velocities = self.motion.calculate_limb_velocity(self.prev_pose, pose)
                angles = self.motion.calculate_joint_angles(pose)
                injury = self.motion.assess_injury_risk(angles, velocities)
                
                if len(self.pose_history) > 10:
                    gait = self.motion.analyze_gait(self.pose_history)
            
            # Draw info panel
            frame = self.draw_info_panel(frame, velocities, angles, injury, gait)
            
            # FPS calculation
            self.frame_count += 1
            if time.time() - self.last_time >= 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = time.time()
            
            cv2.imshow("SportIQ", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.prev_pose = pose
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SportIQ().run()