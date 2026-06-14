import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report
from model import IntrusionDetector
import json

# Load data
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# DataLoader
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Model
model = IntrusionDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
print("Training centralized baseline...")
history = []

for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_preds_binary = (test_preds > 0.5).float()
        f1 = f1_score(y_test_t.numpy(), test_preds_binary.numpy())
        avg_loss = total_loss / len(loader)
        history.append({"epoch": epoch+1, "loss": avg_loss, "f1": f1})
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | F1: {f1:.4f}")

# Final report
model.eval()
with torch.no_grad():
    final_preds = (model(X_test_t) > 0.5).float()
    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(y_test_t.numpy(), final_preds.numpy(),
                                 target_names=["Normal", "Attack"]))

# Save model and history
torch.save(model.state_dict(), "models/baseline_model.pth")
with open("models/baseline_history.json", "w") as f:
    json.dump(history, f)
print("Baseline model saved to models/baseline_model.pth")