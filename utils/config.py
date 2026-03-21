import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config
            return {
                'pose_estimation': {
                    'input_size': [256, 256],
                    'num_keypoints': 17,
                    'backbone': 'yolov8',
                    'pretrained': True
                }
            }
    
    def get(self, key, default=None):
        """Get config value"""
        return self.config.get(key, default)
