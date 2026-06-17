import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import json

CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

class MultiClassIDS(nn.Module):
    def __init__(self, input_dim=41, num_classes=5):
        super(MultiClassIDS, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Load data
X_train = np.load("data/X_train_mc.npy")
X_test  = np.load("data/X_test_mc.npy")
y_train = np.load("data/y_train_mc.npy")
y_test  = np.load("data/y_test_mc.npy")

# SMOTE — oversample minority classes (R2L, U2R)
print("Applying SMOTE to balance classes...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {(y_train == i).sum()}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.FloatTensor(class_weights)
print(f"\nClass weights: {[f'{CLASS_NAMES[i]}:{w:.2f}' for i, w in enumerate(class_weights)]}")

# Tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.LongTensor(y_test)

loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

model     = MultiClassIDS()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

print("\nTraining multi-class IDS...")
history = []

for epoch in range(30):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        f1 = f1_score(y_test_t.numpy(), preds.numpy(), average='macro')
        avg_loss = total_loss / len(loader)
        history.append({"epoch": epoch+1, "loss": avg_loss, "macro_f1": f1})
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Macro F1: {f1:.4f}")

print("\n===== FINAL CLASSIFICATION REPORT =====")
model.eval()
with torch.no_grad():
    final_preds = model(X_test_t).argmax(dim=1)
print(classification_report(y_test_t.numpy(), final_preds.numpy(), target_names=CLASS_NAMES))

torch.save(model.state_dict(), "models/multiclass_model.pth")
with open("models/multiclass_history.json", "w") as f:
    json.dump(history, f)
print("Multi-class model saved!")