import torch
import torch.nn as nn

class IntrusionDetector(nn.Module):
    def __init__(self, input_dim=41):
        super(IntrusionDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return [p.data.clone() for p in self.parameters()]
    
    def set_weights(self, weights):
        for p, w in zip(self.parameters(), weights):
            p.data = w.clone()