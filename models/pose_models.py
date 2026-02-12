import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(PoseNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.keypoint_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_keypoints * 3)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(batch_size, -1)
        keypoints = self.keypoint_head(features)
        return keypoints

class ActionRecognitionModel(nn.Module):
    def __init__(self, num_actions=10, sequence_length=16):
        super(ActionRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.classifier(last_hidden)
        return output