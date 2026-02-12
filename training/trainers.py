import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

class PoseTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('training.learning_rate', 0.001),
            weight_decay=config.get('training.weight_decay', 1e-4)
        )
        
        self.criterion = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch['frames']
            annotations = batch['annotations']
            
            if isinstance(frames, list):
                frames = torch.stack(frames)
            
            frames = frames.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(frames)
            
            loss = self.criterion(predictions, torch.randn_like(predictions))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch['frames']
                if isinstance(frames, list):
                    frames = torch.stack(frames)
                
                frames = frames.to(self.device)
                predictions = self.model(frames)
                
                loss = self.criterion(predictions, torch.randn_like(predictions))
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=100):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss
                }, 'models/best_pose_model.pth')
        
        return best_val_loss

class ActionTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)
    
    def train(self, epochs=50):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                sequences = batch['sequences'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            self.logger.info(f'Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}%')

class BiomechanicsTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.risk_criterion = nn.BCELoss()
        self.performance_criterion = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
    
    def train(self, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                risk_labels = batch['risk_labels'].to(self.device)
                performance_labels = batch['performance_labels'].to(self.device)
                
                self.optimizer.zero_grad()
                risk_pred, performance_pred = self.model(features)
                
                risk_loss = self.risk_criterion(risk_pred.squeeze(), risk_labels)
                performance_loss = self.performance_criterion(performance_pred, performance_labels)
                loss = risk_loss + performance_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            self.logger.info(f'Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}')