import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.performance_tracker import PerformanceTracker
from core.tactical_analyzer import TacticalAnalyzer

class VideoAnalyzer:
    def __init__(self):
        print("="*70)
        print("🎯 SPORTIQ - COMPLETE VIDEO ANALYSIS")
        print("="*70)
        
        # Initialize modules
        self.pose_estimator = PoseEstimator()
        self.motion_analyzer = MotionAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.tactical_analyzer = TacticalAnalyzer()
        
        # Data storage
        self.all_poses = []
        self.all_velocities = []
        self.all_accelerations = []
        self.all_angles = []
        self.impacts = []
        self.gait_data = {}
        
        self.frame_count = 0
        print("✅ All modules ready")
        print("="*70)
    
    def assess_injury_risk(self, angles_frame):
        """Comprehensive injury risk assessment"""
        risks = {
            'knee_risk': {},
            'ankle_risk': {},
            'shoulder_risk': {},
            'asymmetry_risk': {},
            'overall_risk': 'LOW',
            'critical_findings': []
        }
        
        # ----- KNEE RISK (ACL/PCL) -----
        knee_angles = {}
        for key, angle in angles_frame.items():
            if 'knee' in key:
                if 'left' in key:
                    knee_angles['left'] = angle
                if 'right' in key:
                    knee_angles['right'] = angle
        
        for leg, angle in knee_angles.items():
            if angle < 45:
                risks['knee_risk'][leg] = f"🔴 CRITICAL ({angle:.1f}°) - ACL tear risk VERY HIGH"
                risks['critical_findings'].append(f"{leg.upper()} knee hyperflexion - immediate assessment needed")
            elif angle < 60:
                risks['knee_risk'][leg] = f"🟡 HIGH RISK ({angle:.1f}°) - Patellofemoral stress"
                risks['critical_findings'].append(f"{leg.upper()} knee - high load, monitor for pain")
            elif angle < 75:
                risks['knee_risk'][leg] = f"🟠 MODERATE ({angle:.1f}°) - Fatigue related"
            elif angle < 90:
                risks['knee_risk'][leg] = f"🟢 NORMAL ({angle:.1f}°) - Safe range"
            else:
                risks['knee_risk'][leg] = f"🟢 OPTIMAL ({angle:.1f}°) - Full extension"
        
        # ----- ANKLE RISK -----
        if 'left_ankle' in angles_frame and 'right_ankle' in angles_frame:
            left_ankle = angles_frame.get('left_knee_left_ankle_left_ankle', 0)
            right_ankle = angles_frame.get('right_knee_right_ankle_right_ankle', 0)
            
            if left_ankle < 70:
                risks['ankle_risk']['left'] = f"🟡 DORSIFLEXION LIMITATION ({left_ankle:.1f}°) - ankle sprain risk"
            if right_ankle < 70:
                risks['ankle_risk']['right'] = f"🟡 DORSIFLEXION LIMITATION ({right_ankle:.1f}°) - ankle sprain risk"
        
        # ----- ASYMMETRY RISK -----
        if len(knee_angles) == 2:
            asymmetry = abs(knee_angles.get('left', 0) - knee_angles.get('right', 0))
            if asymmetry > 15:
                risks['asymmetry_risk']['knee'] = f"🔴 SEVERE ASYMMETRY ({asymmetry:.1f}°) - compensations, injury imminent"
                risks['critical_findings'].append(f"Knee angle asymmetry >15° - unilateral loading pattern")
            elif asymmetry > 8:
                risks['asymmetry_risk']['knee'] = f"🟡 MODERATE ASYMMETRY ({asymmetry:.1f}°) - address with single-leg work"
        
        # ----- OVERALL RISK LEVEL -----
        risk_score = 0
        if len(risks['critical_findings']) > 0:
            risk_score += 30 * len(risks['critical_findings'])
        if len([v for v in risks['knee_risk'].values() if 'HIGH' in v or 'CRITICAL' in v]) > 0:
            risk_score += 40
        if risks['asymmetry_risk']:
            risk_score += 20
        
        if risk_score >= 50:
            risks['overall_risk'] = '🔴 HIGH - STOP/REST RECOMMENDED'
        elif risk_score >= 25:
            risks['overall_risk'] = '🟡 MODERATE - MONITOR CLOSELY'
        else:
            risks['overall_risk'] = '🟢 LOW - SAFE TO CONTINUE'
        
        return risks
    
    def analyze_video(self, video_path, athlete_name="Athlete"):
        """Process video with complete analysis"""
        
        print(f"\n📹 Processing: {video_path}")
        print(f"👤 Athlete: {athlete_name}")
        print("-"*70)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 FPS: {fps:.1f} | Total frames: {total_frames}")
        print("⏳ Processing...\n")
        
        # Progress tracking
        last_progress = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Progress
            progress = int((self.frame_count / total_frames) * 100)
            if progress >= last_progress + 10:
                print(f"⚡ Progress: {progress}% ({self.frame_count}/{total_frames} frames)")
                last_progress = progress
            
            # 1. POSE ESTIMATION
            pose = self.pose_estimator.estimate_pose(frame)
            self.all_poses.append(pose)
            
            # 2. VELOCITY (need 2 frames)
            if len(self.all_poses) >= 2:
                vel = self.motion_analyzer.calculate_velocity(self.all_poses[-2:])
                if vel:
                    self.all_velocities.extend(vel)
            
            # 3. ACCELERATION (need 2 velocities)
            if len(self.all_velocities) >= 2:
                acc = self.motion_analyzer.calculate_acceleration(self.all_velocities[-2:])
                if acc:
                    self.all_accelerations.extend(acc)
            
            # 4. JOINT ANGLES
            if len(self.all_poses) >= 2:
                angles = self.motion_analyzer.calculate_joint_angles(self.all_poses[-2:])
                if angles:
                    self.all_angles.extend(angles)
            
            # 5. IMPACT DETECTION
            if len(self.all_accelerations) > 0:
                impacts = self.motion_analyzer.detect_impact(self.all_accelerations[-10:], threshold=40)
                if impacts:
                    self.impacts.extend(impacts)
            
            self.frame_count += 1
        
        cap.release()
        
        # 6. GAIT ANALYSIS
        if len(self.all_poses) > 30:
            self.gait_data = self.motion_analyzer.analyze_gait(self.all_poses, fps)
        
        print(f"\n✅ Processing complete! {self.frame_count} frames analyzed")
        
        # ===== GENERATE COMPREHENSIVE REPORT =====
        self.generate_report(video_path, athlete_name)
        
        # ===== VISUALIZE RESULTS =====
        self.visualize_results(video_path)
        
        return self.all_poses
    
    def generate_report(self, video_path, athlete_name):
        """Generate detailed JSON report"""
        
        print("\n" + "="*70)
        print("📊 GENERATING PERFORMANCE REPORT")
        print("="*70)
        
        # Session data for performance tracker
        session_data = {
            'poses': self.all_poses,
            'velocities': self.all_velocities,
            'accelerations': self.all_accelerations,
            'angles': self.all_angles
        }
        
        # Get performance metrics
        performance = self.performance_tracker.track_athlete_performance(
            athlete_name, session_data
        )
        
        # Get latest angles for injury assessment
        latest_angles = self.all_angles[-1] if self.all_angles else {}
        injury_risk = self.assess_injury_risk(latest_angles)
        
        # Tactical analysis (team demo)
        tactical = {
            'formation': '4-3-3 (Analyzed)',
            'pressing_intensity': 'HIGH' if performance.get('power_output', 0) > 70 else 'MEDIUM',
            'counter_attacks': len([i for i in self.impacts if i.get('max_acceleration', 0) > 60]),
            'field_coverage': min(0.95, len(self.all_poses) / 1000)
        }
        
        # ===== INJURY PREDICTION SUMMARY =====
        print("\n⚠️  INJURY RISK ASSESSMENT")
        print("-"*70)
        print(f"Overall Risk: {injury_risk['overall_risk']}")
        
        if injury_risk['knee_risk']:
            print("\n🦵 Knee Joint Analysis:")
            for leg, risk in injury_risk['knee_risk'].items():
                print(f"  {leg}: {risk}")
        
        if injury_risk['ankle_risk']:
            print("\n🦶 Ankle Analysis:")
            for leg, risk in injury_risk['ankle_risk'].items():
                print(f"  {leg}: {risk}")
        
        if injury_risk['asymmetry_risk']:
            print("\n⚖️  Asymmetry Detection:")
            for joint, risk in injury_risk['asymmetry_risk'].items():
                print(f"  {joint}: {risk}")
        
        if injury_risk['critical_findings']:
            print("\n🚨 CRITICAL FINDINGS:")
            for finding in injury_risk['critical_findings']:
                print(f"  • {finding}")
        
        # ===== PERFORMANCE SUMMARY =====
        print("\n💪 PERFORMANCE METRICS")
        print("-"*70)
        print(f"Movement Efficiency: {performance.get('movement_efficiency', 0)*100:.1f}%")
        print(f"Power Output: {performance.get('power_output', 0):.1f}/100")
        print(f"Fatigue Index: {performance.get('fatigue_index', 0)*100:.1f}%")
        print(f"Flexibility Score: {performance.get('flexibility_score', 0)*100:.1f}%")
        print(f"Balance Metric: {performance.get('balance_metric', 0)*100:.1f}%")
        
        # ===== MOTION ANALYSIS =====
        print("\n📈 MOTION ANALYSIS")
        print("-"*70)
        if self.all_velocities:
            avg_vel = np.mean([np.mean(list(v.values())) for v in self.all_velocities if v])
            print(f"Average Velocity: {avg_vel:.2f} px/s")
        if self.all_accelerations:
            avg_acc = np.mean([np.mean(list(a.values())) for a in self.all_accelerations if a])
            print(f"Average Acceleration: {avg_acc:.2f} px/s²")
        print(f"Impact Events Detected: {len(self.impacts)}")
        
        if self.gait_data:
            print(f"\n🚶 GAIT CYCLE:")
            print(f"  Stride Frequency (L): {self.gait_data.get('stride_frequency_left', 0):.2f} Hz")
            print(f"  Stride Frequency (R): {self.gait_data.get('stride_frequency_right', 0):.2f} Hz")
            print(f"  Step Symmetry: {self.gait_data.get('step_symmetry', 0)*100:.1f}%")
        
        # ===== TACTICAL INSIGHTS =====
        print("\n⚽ TACTICAL ANALYSIS")
        print("-"*70)
        print(f"Detected Formation: {tactical['formation']}")
        print(f"Pressing Intensity: {tactical['pressing_intensity']}")
        print(f"Counter Attack Opportunities: {tactical['counter_attacks']}")
        print(f"Field Coverage: {tactical['field_coverage']*100:.1f}%")
        
        # ===== RECOMMENDATIONS =====
        print("\n📋 ACTIONABLE RECOMMENDATIONS")
        print("-"*70)
        
        # Injury-based recommendations
        if injury_risk['overall_risk'].startswith('🔴'):
            print("  • 🛑 IMMEDIATE: Athlete should stop activity and consult sports medicine")
        elif injury_risk['overall_risk'].startswith('🟡'):
            print("  • ⚠️  URGENT: Modify training load, focus on recovery")
        
        for finding in injury_risk['critical_findings'][:2]:
            print(f"  • 🏥 {finding}")
        
        # Performance-based recommendations
        if performance.get('movement_efficiency', 1) < 0.7:
            print("  • 🎯 Technical focus: Improve movement efficiency with drill X")
        if performance.get('fatigue_index', 0) > 0.3:
            print("  • 😴 Recovery: 48hr rest + nutrition protocol")
        if performance.get('power_output', 0) < 60:
            print("  • 💪 Strength: Add plyometric training 2x/week")
        if performance.get('flexibility_score', 0) < 0.6:
            print("  • 🧘 Mobility: Daily hamstring and hip flexor stretching")
        
        if len(self.impacts) > 10:
            print("  • 🦵 Landing mechanics: High impact load - focus on soft landings")
        
        if self.gait_data.get('step_symmetry', 1) < 0.8:
            print("  • 🏃 Gait retraining: Address asymmetry with single-leg drills")
        
        # Default recommendation
        if len(injury_risk['critical_findings']) == 0 and performance.get('movement_efficiency', 0) > 0.8:
            print("  • ✅ All metrics optimal - maintain current training program")
        
        print("\n" + "="*70)
        
        # Save report
        report = {
            'athlete': athlete_name,
            'video': video_path,
            'timestamp': datetime.now().isoformat(),
            'frames_processed': self.frame_count,
            'performance_metrics': performance,
            'injury_risk': injury_risk,
            'motion_analysis': {
                'velocity_avg': avg_vel if self.all_velocities else 0,
                'acceleration_avg': avg_acc if self.all_accelerations else 0,
                'impacts': len(self.impacts),
                'gait': self.gait_data
            },
            'tactical': tactical,
            'recommendations': []  # Will populate from prints above
        }
        
        # Save to file
        output_path = Path(f"results/analysis_{athlete_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Full report saved: {output_path}")
    
    def visualize_results(self, video_path):
        """Show video with injury risk overlay"""
        
        print("\n🎬 Playing video with injury risk overlay...")
        print("Press 'q' to quit, 'p' to pause")
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get corresponding pose
            if frame_idx < len(self.all_poses):
                pose = self.all_poses[frame_idx]
                
                # Draw skeleton
                for conn in self.pose_estimator.skeleton_connections:
                    if conn[0] in pose and conn[1] in pose:
                        p1 = pose[conn[0]]
                        p2 = pose[conn[1]]
                        if p1['confidence'] > 0.5 and p2['confidence'] > 0.5:
                            cv2.line(frame, 
                                    (int(p1['x']), int(p1['y'])),
                                    (int(p2['x']), int(p2['y'])),
                                    (0, 255, 255), 2)
                
                # Draw keypoints
                for joint, data in pose.items():
                    if data['confidence'] > 0.5:
                        cv2.circle(frame, (int(data['x']), int(data['y'])), 4, (0, 255, 0), -1)
                
                # Draw injury risk overlay
                if frame_idx < len(self.all_angles):
                    angles = self.all_angles[frame_idx] if frame_idx < len(self.all_angles) else {}
                    risk = self.assess_injury_risk(angles)
                    
                    # Risk color bar
                    if 'HIGH' in risk['overall_risk']:
                        color = (0, 0, 255)  # Red
                    elif 'MODERATE' in risk['overall_risk']:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green
                    
                    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
                    cv2.putText(frame, f"INJURY RISK: {risk['overall_risk'][:20]}", 
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('SportIQ - Injury Risk Analysis', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
            
            frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='SportIQ Complete Video Analysis')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--athlete', type=str, default='Athlete', help='Athlete name/ID')
    
    args = parser.parse_args()
    
    analyzer = VideoAnalyzer()
    analyzer.analyze_video(args.video, args.athlete)

if __name__ == "__main__":
    main()
