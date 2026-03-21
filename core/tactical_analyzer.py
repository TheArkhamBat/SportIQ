import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

class TacticalAnalyzer:
    def __init__(self):
        self.team_positions = defaultdict(list)
        self.player_trajectories = defaultdict(list)
    
    def analyze_formation(self, frame_data, team_assignment):
        team1_positions = []
        team2_positions = []
        
        for player_id, pose in frame_data.items():
            if player_id in team_assignment:
                team = team_assignment[player_id]
                position = self._get_player_position(pose)
                
                if team == 1:
                    team1_positions.append(position)
                else:
                    team2_positions.append(position)
        
        formation1 = self._detect_formation(team1_positions) if team1_positions else {}
        formation2 = self._detect_formation(team2_positions) if team2_positions else {}
        
        return {
            'team1_formation': formation1,
            'team2_formation': formation2
        }
    
    def _get_player_position(self, pose):
        if pose['left_hip']['confidence'] > 0.5 and pose['right_hip']['confidence'] > 0.5:
            x = (pose['left_hip']['x'] + pose['right_hip']['x']) / 2
            y = (pose['left_hip']['y'] + pose['right_hip']['y']) / 2
            return (x, y)
        elif pose['nose']['confidence'] > 0.5:
            return (pose['nose']['x'], pose['nose']['y'])
        else:
            return (0.5, 0.5)
    
    def _detect_formation(self, positions):
        if len(positions) < 3:
            return {'type': 'unknown', 'confidence': 0}
        
        positions_array = np.array(positions)
        
        clustering = DBSCAN(eps=0.2, min_samples=2).fit(positions_array)
        clusters = {}
        
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(positions_array[i])
        
        cluster_centers = []
        for cluster_points in clusters.values():
            if len(cluster_points) > 1:
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append(center)
        
        formation_type = self._classify_formation(cluster_centers)
        
        return {
            'type': formation_type,
            'cluster_centers': [center.tolist() for center in cluster_centers],
            'confidence': min(len(cluster_centers) / 5, 1.0)
        }
    
    def _classify_formation(self, cluster_centers):
        if not cluster_centers:
            return 'unknown'
        
        num_clusters = len(cluster_centers)
        
        if num_clusters == 1:
            return '4-4-2'
        elif num_clusters == 2:
            return '4-3-3'
        elif num_clusters >= 3:
            return '3-5-2'
        else:
            return 'custom'
    
    def track_player_movements(self, player_poses_sequence):
        trajectories = {}
        
        for player_id, poses in player_poses_sequence.items():
            trajectory = []
            for pose in poses:
                position = self._get_player_position(pose)
                trajectory.append(position)
            trajectories[player_id] = trajectory
        
        return trajectories
    
    def calculate_field_coverage(self, trajectories, field_dimensions=(1.0, 1.0)):
        all_positions = []
        for trajectory in trajectories.values():
            all_positions.extend(trajectory)
        
        if not all_positions:
            return 0
        
        positions_array = np.array(all_positions)
        
        x_min, y_min = np.min(positions_array, axis=0)
        x_max, y_max = np.max(positions_array, axis=0)
        
        coverage_width = x_max - x_min
        coverage_height = y_max - y_min
        
        total_coverage = coverage_width * coverage_height
        field_area = field_dimensions[0] * field_dimensions[1]
        
        return total_coverage / field_area
    
    def detect_tactical_patterns(self, trajectories, team_assignment):
        patterns = {
            'pressing': self._detect_pressing(trajectories, team_assignment),
            'counter_attack': self._detect_counter_attack(trajectories, team_assignment),
            'possession_play': self._detect_possession_play(trajectories, team_assignment)
        }
        
        return patterns
    
    def _detect_pressing(self, trajectories, team_assignment):
        if len(trajectories) < 4:
            return {'detected': False, 'intensity': 0}
        
        team1_avg_pos = self._get_team_average_position(trajectories, team_assignment, 1)
        team2_avg_pos = self._get_team_average_position(trajectories, team_assignment, 2)
        
        if team1_avg_pos is None or team2_avg_pos is None:
            return {'detected': False, 'intensity': 0}
        
        distance = np.linalg.norm(np.array(team1_avg_pos) - np.array(team2_avg_pos))
        pressing_intensity = max(0, 1 - distance)
        
        return {
            'detected': pressing_intensity > 0.6,
            'intensity': pressing_intensity
        }
    
    def _detect_counter_attack(self, trajectories, team_assignment):
        if len(trajectories) < 4:
            return {'detected': False, 'speed': 0}
        
        team_velocities = self._calculate_team_velocities(trajectories, team_assignment)
        
        max_velocity = max(team_velocities.values()) if team_velocities else 0
        
        return {
            'detected': max_velocity > 0.8,
            'speed': max_velocity
        }
    
    def _detect_possession_play(self, trajectories, team_assignment):
        field_coverage = self.calculate_field_coverage(trajectories)
        
        return {
            'detected': field_coverage > 0.7,
            'coverage': field_coverage
        }
    
    def _get_team_average_position(self, trajectories, team_assignment, team_id):
        team_positions = []
        
        for player_id, trajectory in trajectories.items():
            if player_id in team_assignment and team_assignment[player_id] == team_id:
                if trajectory:
                    team_positions.append(trajectory[-1])
        
        if not team_positions:
            return None
        
        return np.mean(team_positions, axis=0)
    
    def _calculate_team_velocities(self, trajectories, team_assignment):
        team_velocities = {}
        
        for team_id in [1, 2]:
            team_trajectories = []
            for player_id, trajectory in trajectories.items():
                if player_id in team_assignment and team_assignment[player_id] == team_id:
                    team_trajectories.extend(trajectory)
            
            if len(team_trajectories) > 1:
                velocities = []
                for i in range(1, len(team_trajectories)):
                    dist = np.linalg.norm(np.array(team_trajectories[i]) - np.array(team_trajectories[i-1]))
                    velocities.append(dist)
                
                team_velocities[team_id] = np.mean(velocities) if velocities else 0
        
        return team_velocities
    
    def generate_tactical_report(self, match_data):
        report = {
            'formations': self.analyze_formation(match_data['frame_data'], match_data['team_assignment']),
            'field_coverage': self.calculate_field_coverage(match_data['trajectories']),
            'tactical_patterns': self.detect_tactical_patterns(match_data['trajectories'], match_data['team_assignment']),
            'recommendations': []
        }
        
        self._generate_tactical_recommendations(report)
        return report
    
    def _generate_tactical_recommendations(self, report):
        patterns = report['tactical_patterns']
        
        if patterns['pressing']['detected'] and patterns['pressing']['intensity'] > 0.8:
            report['recommendations'].append("Opponent using high press - consider long balls")
        
        if patterns['counter_attack']['detected']:
            report['recommendations'].append("Strong counter-attacking threat - maintain defensive shape")
        
        if patterns['possession_play']['detected']:
            report['recommendations'].append("Opponent dominating possession - focus on pressing triggers")
        
        if report['field_coverage'] < 0.6:
            report['recommendations'].append("Low field coverage - improve positional play")