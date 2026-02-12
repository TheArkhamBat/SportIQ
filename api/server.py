from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import os
import sys
from datetime import datetime

sys.path.append('..')
from core.pose_estimator import PoseEstimator
from core.motion_analyzer import MotionAnalyzer
from core.performance_tracker import PerformanceTracker
from core.tactical_analyzer import TacticalAnalyzer
from utils.visualization import VisualizationEngine
from utils.helpers import setup_logging

app = Flask(__name__)
logger = setup_logging()

pose_estimator = None
motion_analyzer = None
performance_tracker = None
tactical_analyzer = None
visualization_engine = None

def initialize_services():
    global pose_estimator, motion_analyzer, performance_tracker, tactical_analyzer, visualization_engine
    
    try:
        pose_estimator = PoseEstimator('./models/pose_model.pth')
        motion_analyzer = MotionAnalyzer()
        performance_tracker = PerformanceTracker()
        tactical_analyzer = TacticalAnalyzer()
        visualization_engine = VisualizationEngine()
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'SportIQ API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        athlete_id = request.form.get('athlete_id', 'default_athlete')
        analysis_type = request.form.get('analysis_type', 'full')
        
        video_path = f'./temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
        video_file.save(video_path)
        
        if pose_estimator is None:
            initialize_services()
        
        poses = pose_estimator.process_video(video_path)
        
        if analysis_type in ['motion', 'full']:
            velocities = motion_analyzer.calculate_velocity(poses)
            accelerations = motion_analyzer.calculate_acceleration(velocities)
            joint_angles = motion_analyzer.calculate_joint_angles(poses)
            impacts = motion_analyzer.detect_impact(accelerations)
            gait_analysis = motion_analyzer.analyze_gait(poses)
        else:
            velocities, accelerations, joint_angles, impacts, gait_analysis = [], [], [], [], {}
        
        if analysis_type in ['performance', 'full']:
            session_data = {
                'poses': poses,
                'velocities': velocities,
                'accelerations': accelerations,
                'angles': joint_angles
            }
            performance_metrics = performance_tracker.track_athlete_performance(athlete_id, session_data)
        else:
            performance_metrics = {}
        
        os.remove(video_path)
        
        result = {
            'athlete_id': athlete_id,
            'analysis_type': analysis_type,
            'poses_detected': len(poses),
            'motion_analysis': {
                'velocities': len(velocities),
                'accelerations': len(accelerations),
                'joint_angles': len(joint_angles),
                'impacts_detected': len(impacts),
                'gait_analysis': gait_analysis
            },
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/tactical', methods=['POST'])
def analyze_tactical():
    try:
        data = request.get_json()
        
        if not data or 'frame_data' not in data or 'team_assignment' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        frame_data = data['frame_data']
        team_assignment = data['team_assignment']
        
        if tactical_analyzer is None:
            initialize_services()
        
        formation_analysis = tactical_analyzer.analyze_formation(frame_data, team_assignment)
        
        if 'trajectories' in data:
            trajectories = data['trajectories']
            tactical_patterns = tactical_analyzer.detect_tactical_patterns(trajectories, team_assignment)
            field_coverage = tactical_analyzer.calculate_field_coverage(trajectories)
        else:
            tactical_patterns = {}
            field_coverage = 0
        
        match_data = {
            'frame_data': frame_data,
            'team_assignment': team_assignment,
            'trajectories': trajectories if 'trajectories' in data else {}
        }
        
        tactical_report = tactical_analyzer.generate_tactical_report(match_data)
        
        return jsonify(tactical_report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize/pose', methods=['POST'])
def visualize_pose():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if pose_estimator is None:
            initialize_services()
        
        pose = pose_estimator.estimate_pose(image)
        
        visualized_image = visualization_engine.draw_pose_on_frame(image, pose)
        
        output_path = f'./output_pose_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(output_path, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
        
        return send_file(output_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance/report', methods=['GET'])
def get_performance_report():
    try:
        athlete_id = request.args.get('athlete_id', 'default_athlete')
        
        if performance_tracker is None:
            initialize_services()
        
        report = performance_tracker.generate_performance_report(athlete_id)
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/biomechanics/analyze', methods=['POST'])
def analyze_biomechanics():
    try:
        data = request.get_json()
        
        if not data or 'poses' not in data:
            return jsonify({'error': 'No pose data provided'}), 400
        
        poses = data['poses']
        
        if motion_analyzer is None:
            initialize_services()
        
        velocities = motion_analyzer.calculate_velocity(poses)
        accelerations = motion_analyzer.calculate_acceleration(velocities)
        joint_angles = motion_analyzer.calculate_joint_angles(poses)
        impacts = motion_analyzer.detect_impact(accelerations)
        gait_analysis = motion_analyzer.analyze_gait(poses)
        
        biomechanics_data = {
            'velocities': velocities,
            'accelerations': accelerations,
            'joint_angles': joint_angles,
            'impacts': impacts,
            'gait_analysis': gait_analysis
        }
        
        return jsonify(biomechanics_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_services()
    app.run(host='0.0.0.0', port=8000, debug=True)