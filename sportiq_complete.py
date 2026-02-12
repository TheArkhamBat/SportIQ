import cv2
import numpy as np
import time
from datetime import datetime

from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.performance_tracker import PerformanceTracker
from core.tactical_analyzer import TacticalAnalyzer
from utils.visualization import VisualizationEngine

class SportIQComplete:
    def __init__(self):
        print("="*70)
        print("🏆 SPORTIQ - COMPLETE ATHLETIC PERFORMANCE ANALYTICS")
        print("="*70)
        
        # Initialize all modules
        self.pose_estimator = PoseEstimator()
        self.motion_analyzer = MotionAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.tactical_analyzer = TacticalAnalyzer()
        self.viz = VisualizationEngine()
        
        # Store data for analysis
        self.pose_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.angle_history = []
        
        # Performance metrics
        self.athlete_id = f"athlete_{datetime.now().strftime('%H%M%S')}"
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        print("✅ All modules initialized")
        print("="*70)
    
    def analyze_frame(self, pose):
        """Run all analysis on single frame"""
        analysis = {
            'motion': {},
            'performance': {},
            'tactical': {},
            'injury_risk': {},
            'recommendations': []
        }
        
        # Store pose for history
        self.pose_history.append(pose)
        if len(self.pose_history) > 30:  # Keep last 30 frames
            self.pose_history.pop(0)
        
        # Need at least 2 frames for velocity
        if len(self.pose_history) >= 2:
            # Calculate velocity
            velocity = self.motion_analyzer.calculate_velocity(self.pose_history[-2:])
            if velocity:
                self.velocity_history.extend(velocity)
                analysis['motion']['velocity'] = np.mean([v for frame_v in velocity[-5:] for v in frame_v.values()]) if velocity else 0
            
            # Calculate acceleration
            if len(self.velocity_history) >= 2:
                acceleration = self.motion_analyzer.calculate_acceleration(self.velocity_history[-2:])
                if acceleration:
                    self.acceleration_history.extend(acceleration)
                    analysis['motion']['acceleration'] = np.mean([a for frame_a in acceleration[-5:] for a in frame_a.values()]) if acceleration else 0
            
            # Calculate joint angles
            angles = self.motion_analyzer.calculate_joint_angles(self.pose_history[-2:])
            if angles:
                self.angle_history.extend(angles)
                
                # INJURY RISK ASSESSMENT
                analysis['injury_risk'] = self.assess_injury_risk(angles[-1] if angles else {})
        
        # PERFORMANCE METRICS
        if len(self.velocity_history) > 5:
            session_data = {
                'poses': self.pose_history,
                'velocities': self.velocity_history[-30:],
                'accelerations': self.acceleration_history[-30:],
                'angles': self.angle_history[-30:]
            }
            performance = self.performance_tracker.track_athlete_performance(
                self.athlete_id, session_data
            )
            analysis['performance'] = performance
        
        # TACTICAL ANALYSIS (for team sports - single athlete demo)
        analysis['tactical'] = {
            'formation': '4-3-3 (Demo)',
            'pressing_intensity': 'HIGH' if analysis['motion'].get('velocity', 0) > 5 else 'MEDIUM',
            'field_coverage': 0.73
        }
        
        # GENERATE RECOMMENDATIONS
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def assess_injury_risk(self, angles):
        """Real-time injury risk assessment"""
        risks = {}
        
        # Knee angle risk (ACL)
        for leg in ['left', 'right']:
            knee_key = f'{leg}_knee_left_hip_{leg}_knee_{leg}_ankle'
            if knee_key in angles:
                angle = angles[knee_key]
                if angle < 45:
                    risks[f'{leg}_knee'] = f"🔴 HIGH RISK ({angle:.1f}°) - ACL strain possible"
                elif angle < 70:
                    risks[f'{leg}_knee'] = f"🟡 MODERATE RISK ({angle:.1f}°) - Monitor fatigue"
                else:
                    risks[f'{leg}_knee'] = f"🟢 LOW RISK ({angle:.1f}°) - Optimal"
        
        # Elbow angle risk
        for arm in ['left', 'right']:
            elbow_key = f'{arm}_shoulder_{arm}_elbow_{arm}_wrist'
            if elbow_key in angles:
                angle = angles[elbow_key]
                if 160 <= angle <= 180:
                    risks[f'{arm}_elbow'] = f"🟢 OPTIMAL ({angle:.1f}°)"
                elif angle < 160:
                    risks[f'{arm}_elbow'] = f"🟡 HYPEREXTENSION RISK ({angle:.1f}°)"
                else:
                    risks[f'{arm}_elbow'] = f"🟡 POOR FORM ({angle:.1f}°)"
        
        return risks
    
    def generate_recommendations(self, analysis):
        """Generate actionable insights"""
        recs = []
        
        # Injury-based recommendations
        for joint, risk in analysis.get('injury_risk', {}).items():
            if 'HIGH' in risk:
                recs.append(f"⚠️ {joint}: Rest recommended - high injury risk")
            elif 'MODERATE' in risk:
                recs.append(f"⚡ {joint}: Monitor fatigue levels")
        
        # Performance-based recommendations
        perf = analysis.get('performance', {})
        if perf.get('movement_efficiency', 1) < 0.7:
            recs.append("📉 Low movement efficiency - focus on technique")
        if perf.get('fatigue_index', 0) > 0.3:
            recs.append("😓 High fatigue detected - consider rest")
        if perf.get('power_output', 0) < 60:
            recs.append("💪 Power output low - explosive training recommended")
        
        # Tactical recommendations
        if analysis.get('tactical', {}).get('pressing_intensity') == 'LOW':
            recs.append("⚽ Increase pressing intensity - opponent has time")
        
        if not recs:
            recs.append("✅ All metrics optimal - maintain current training")
        
        return recs[:3]  # Top 3 recommendations
    
    def draw_analysis(self, frame, analysis):
        """Draw all analysis metrics on frame"""
        h, w = frame.shape[:2]
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "SPORTIQ - COMPLETE ANALYSIS", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y = 70
        # FPS and Frame
        cv2.putText(frame, f"FPS: {self.fps:.1f} | Frame: {self.frame_count}", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        
        # Motion Analysis
        motion = analysis.get('motion', {})
        cv2.putText(frame, f"⚡ Velocity: {motion.get('velocity', 0):.1f} px/s", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 20
        cv2.putText(frame, f"📈 Acceleration: {motion.get('acceleration', 0):.1f} px/s²", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y += 25
        
        # Performance Metrics
        perf = analysis.get('performance', {})
        cv2.putText(frame, f"💪 Power: {perf.get('power_output', 0):.0f}/100", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y += 20
        cv2.putText(frame, f"🎯 Efficiency: {perf.get('movement_efficiency', 0)*100:.0f}%", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y += 20
        cv2.putText(frame, f"😓 Fatigue: {perf.get('fatigue_index', 0)*100:.0f}%", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y += 25
        
        # Injury Risk
        cv2.putText(frame, "⚠️ INJURY RISK:", (20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 20
        for joint, risk in list(analysis.get('injury_risk', {}).items())[:2]:
            short_risk = risk[:30] + "..." if len(risk) > 30 else risk
            cv2.putText(frame, f"  {short_risk}", (20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 18
        
        y += 5
        # Tactical
        tactical = analysis.get('tactical', {})
        cv2.putText(frame, f"⚽ Formation: {tactical.get('formation', 'N/A')}", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        y += 20
        cv2.putText(frame, f"🔄 Pressing: {tactical.get('pressing_intensity', 'N/A')}", 
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        y += 25
        
        # Recommendations
        cv2.putText(frame, "📋 RECOMMENDATIONS:", (20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 20
        for i, rec in enumerate(analysis.get('recommendations', [])):
            cv2.putText(frame, f"  {i+1}. {rec}", (20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 18
        
        return frame
    
    def run(self):
        """Main realtime loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎥 LIVE ANALYSIS - Press 'q' to quit")
        print("="*70)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS calculation
            self.frame_count += 1
            if time.time() - self.fps_time >= 1:
                self.fps = self.frame_count
                self.frame_count = 0
                self.fps_time = time.time()
            
            # Pose estimation
            pose = self.pose_estimator.estimate_pose(frame)
            
            # Complete analysis
            analysis = self.analyze_frame(pose)
            
            # Draw pose skeleton
            frame = self.viz.draw_pose_on_frame(frame, pose)
            
            # Draw analysis overlay
            frame = self.draw_analysis(frame, analysis)
            
            # Show frame
            cv2.imshow('SportIQ - Complete Athletic Analytics', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Analysis stopped")
        print("="*70)

if __name__ == "__main__":
    app = SportIQComplete()
    app.run()
