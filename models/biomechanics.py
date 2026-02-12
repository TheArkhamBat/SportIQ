import torch
import torch.nn as nn
import numpy as np

class BiomechanicsModel(nn.Module):
    def __init__(self, input_dim=51, hidden_dims=[256, 128, 64]):
        super(BiomechanicsModel, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.risk_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.performance_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        injury_risk = self.risk_predictor(features)
        performance_scores = self.performance_predictor(features)
        return injury_risk, performance_scores

class GaitAnalysisModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(GaitAnalysisModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.classifier(last_output)
        return output