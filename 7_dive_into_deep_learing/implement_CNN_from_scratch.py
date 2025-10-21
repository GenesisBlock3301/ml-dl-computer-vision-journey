import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# -----------------------------
# 1. Simple synthetic "image" dataset
# -----------------------------
# Let's make simple grayscale 8x8 images from blobs

def make_synthetic_dataset(n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=n_samples, centers=4, n_features=2, random_state=seed)
    # normalize to [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    imgs = np.zeros((n_samples, 1, 8, 8), dtype=np.float32)

    for i, (a, b) in enumerate(X):
        x_idx = min(int(a * 7), 7)  # ensure index in [0,7]
        y_idx = min(int(b * 7), 7)
        imgs[i, 0, x_idx, y_idx] = 1.0
        imgs[i, 0] += rng.random((8, 8), dtype=np.float32) * 0.1
        imgs[i, 0] = np.clip(imgs[i, 0], 0.0, 1.0)

    return torch.tensor(imgs, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X, y = make_synthetic_dataset(600)
train_ds = TensorDataset(X, y)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# -----------------------------
# 2. CNN Model definition
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, activation="relu", dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 🔧 fixed size to avoid shape mismatch
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
        self.fc1 = nn.Linear(32 * 4 * 4, 64)  # ✅ shape fixed
        self.fc2 = nn.Linear(64, 4)

        if activation == "relu":
            self.act = F.relu
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)

# -----------------------------
# 3. Train Function
# -----------------------------
def train_model(model, loss_fn, optimizer, epochs=10):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_dl))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
    return losses

# -----------------------------
# 4. Train with different activations & loss
# -----------------------------
configs = [
    ("relu", "CrossEntropyLoss"),
    ("sigmoid", "CrossEntropyLoss"),
    ("tanh", "CrossEntropyLoss"),
]

for act, loss_name in configs:
    print(f"\n🔹 Training with activation={act}, loss={loss_name}")
    model = SimpleCNN(activation=act)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 Regularization
    losses = train_model(model, loss_fn, optimizer, epochs=8)

    plt.plot(losses, label=f"{act}")
plt.title("Loss curve with different activations")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -----------------------------
# 5. Dropout vs No Dropout (Overfitting demo)
# -----------------------------
print("\n🔹 Overfitting demo (dropout vs no dropout)")
small_data = list(zip(X[:100], y[:100]))  # intentionally small dataset
small_dl = DataLoader(TensorDataset(X[:100], y[:100]), batch_size=16, shuffle=True)

def overfit_demo(dropout):
    model = SimpleCNN(activation="relu", dropout=dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    losses = []
    for _ in range(12):
        epoch_loss = 0
        for xb, yb in small_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(small_dl))
    return losses

no_drop = overfit_demo(dropout=False)
with_drop = overfit_demo(dropout=True)

plt.plot(no_drop, label="No Dropout (Overfit)")
plt.plot(with_drop, label="With Dropout (Regularized)")
plt.title("Dropout regularization effect")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
