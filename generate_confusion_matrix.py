import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

class MultiClassIDS(nn.Module):
    def __init__(self, input_dim=41, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

X_test = np.load("data/X_test_mc.npy")
y_test = np.load("data/y_test_mc.npy")

model = MultiClassIDS()
model.load_state_dict(torch.load("models/federated_multiclass_model.pth"))
model.eval()

with torch.no_grad():
    preds = model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('FedShield — Multi-Class Attack Detection\nConfusion Matrix (Federated Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("Saved confusion matrix to models/confusion_matrix.png")
plt.show()