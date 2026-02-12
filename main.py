import argparse
import torch
import numpy as np
import logging
import os

from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.performance_tracker import PerformanceTracker
from core.tactical_analyzer import TacticalAnalyzer
from utils.config import Config
from utils.helpers import setup_logging, create_directories
from training.trainers import PoseTrainer, ActionTrainer

class SportIQ:
    def __init__(self, config_path='config/model_config.yaml'):
        self.config = Config(config_path)
        self.logger = setup_logging()
        create_directories()
        
        self.pose_estimator = None
        self.motion_analyzer = None
        self.performance_tracker = None
        self.tactical_analyzer = None
        
    def initialize_services(self):
        self.logger.info("Initializing SportIQ services...")
        
        self.pose_estimator = PoseEstimator()
        self.motion_analyzer = MotionAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.tactical_analyzer = TacticalAnalyzer()
        
        self.logger.info("All services initialized successfully")
    
    def analyze_video(self, video_path, athlete_id="default", analysis_type="full"):
        self.logger.info(f"Analyzing video: {video_path}")
        
        if self.pose_estimator is None:
            self.initialize_services()
        
        poses = self.pose_estimator.process_video(video_path)
        
        if analysis_type in ["motion", "full"]:
            velocities = self.motion_analyzer.calculate_velocity(poses)
            accelerations = self.motion_analyzer.calculate_acceleration(velocities)
            joint_angles = self.motion_analyzer.calculate_joint_angles(poses)
            impacts = self.motion_analyzer.detect_impact(accelerations)
            gait_analysis = self.motion_analyzer.analyze_gait(poses)
        else:
            velocities, accelerations, joint_angles, impacts, gait_analysis = [], [], [], [], {}
        
        if analysis_type in ["performance", "full"]:
            session_data = {
                'poses': poses,
                'velocities': velocities,
                'accelerations': accelerations,
                'angles': joint_angles
            }
            performance_metrics = self.performance_tracker.track_athlete_performance(athlete_id, session_data)
        else:
            performance_metrics = {}
        
        result = {
            'video_path': video_path,
            'athlete_id': athlete_id,
            'analysis_type': analysis_type,
            'poses_detected': len(poses),
            'motion_analysis': {
                'velocities': velocities,
                'accelerations': accelerations,
                'joint_angles': joint_angles,
                'impacts': impacts,
                'gait_analysis': gait_analysis
            },
            'performance_metrics': performance_metrics
        }
        
        return result
    
    def train_models(self):
        self.logger.info("Training SportIQ models...")
        
        from models.pose_models import PoseNet
        from data.dataloader import SportsDataLoader
        
        model = PoseNet()
        data_loader = SportsDataLoader(self.config)
        
        train_loader = data_loader.create_data_loader('./data/train')
        val_loader = data_loader.create_data_loader('./data/val')
        
        trainer = PoseTrainer(model, train_loader, val_loader, self.config)
        best_loss = trainer.train()
        
        self.logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
    
    def run_real_time_analysis(self, camera_id=0):
        self.logger.info(f"Starting real-time analysis from camera {camera_id}")
        
        if self.pose_estimator is None:
            self.initialize_services()
        
        import cv2
        
        cap = cv2.VideoCapture(camera_id)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose = self.pose_estimator.estimate_pose(frame)
            
            from utils.visualization import VisualizationEngine
            viz_engine = VisualizationEngine()
            visualized_frame = viz_engine.draw_pose_on_frame(frame, pose)
            
            cv2.imshow('SportIQ Real-Time Analysis', visualized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='SportIQ: Athletic Performance Analytics')
    parser.add_argument('--mode', choices=['analyze', 'train', 'realtime', 'api'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--video', type=str, help='Video file path for analysis')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for real-time analysis')
    parser.add_argument('--athlete', type=str, default='default', help='Athlete ID')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    sportiq = SportIQ(args.config)
    
    if args.mode == 'analyze':
        if not args.video:
            print("Please provide a video file with --video")
            return
        
        result = sportiq.analyze_video(args.video, args.athlete)
        print(f"Analysis completed for {args.video}")
        print(f"Poses detected: {result['poses_detected']}")
        
    elif args.mode == 'train':
        sportiq.train_models()
        
    elif args.mode == 'realtime':
        sportiq.run_real_time_analysis(args.camera)
        
    elif args.mode == 'api':
        from api.server import app
        app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == '__main__':
    main()