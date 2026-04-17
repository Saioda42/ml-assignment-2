#!/usr/bin/env python
"""
Retrain MLP with improved regularization to prevent overfitting:
- Increased dropout from 0.3 to 0.5
- Reduced hidden sizes from [128, 64] to [64, 32]
- Added L2 regularization (weight_decay)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ucimlrepo import fetch_ucirepo

# ==================== LOAD DATA ====================
print("Loading data...")
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets
X = X.drop('education', axis=1)

# ==================== PREPROCESS ====================
print("Preprocessing...")
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Handle missing values
workclass_mode = X_train['workclass'].mode()[0]
occupation_mode = X_train['occupation'].mode()[0]
country_mode = X_train['native-country'].mode()[0]

for X_set in [X_train, X_val, X_test]:
    X_set['workclass'] = X_set['workclass'].fillna(workclass_mode)
    X_set['occupation'] = X_set['occupation'].fillna(occupation_mode)
    X_set['native-country'] = X_set['native-country'].fillna(country_mode)

# Encode categorical features
cat_cols_list = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
num_cols_list = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

for col in cat_cols_list:
    X_train[col] = X_train[col].astype(str)
    X_val[col] = X_val[col].astype(str)
    X_test[col] = X_test[col].astype(str)

label_encoders = {}
for col in cat_cols_list:
    le = LabelEncoder()
    all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).unique()
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Scale numerical features
for col in ['capital-gain', 'capital-loss']:
    X_train[col] = np.log1p(X_train[col])
    X_val[col] = np.log1p(X_val[col])
    X_test[col] = np.log1p(X_test[col])

scaler = StandardScaler()
X_train[num_cols_list] = scaler.fit_transform(X_train[num_cols_list])
X_val[num_cols_list] = scaler.transform(X_val[num_cols_list])
X_test[num_cols_list] = scaler.transform(X_test[num_cols_list])

# ==================== PREPARE TORCH DATASET ====================
class AdultDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = X_cat
        self.X_num = X_num
        self.y = y

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_cat[idx], dtype=torch.long),
            torch.tensor(self.X_num[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

X_train_cat = X_train[cat_cols_list].values
X_train_num = X_train[num_cols_list].values
X_val_cat = X_val[cat_cols_list].values
X_val_num = X_val[num_cols_list].values
X_test_cat = X_test[cat_cols_list].values
X_test_num = X_test[num_cols_list].values

y['income'] = y['income'].str.strip('.')
y_train_encoded = (y_train == '>50K').astype(int).values
y_val_encoded = (y_val == '>50K').astype(int).values
y_test_encoded = (y_test == '>50K').astype(int).values

train_dataset = AdultDataset(X_train_cat, X_train_num, y_train_encoded)
val_dataset = AdultDataset(X_val_cat, X_val_num, y_val_encoded)
test_dataset = AdultDataset(X_test_cat, X_test_num, y_test_encoded)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== SETUP EMBEDDINGS ====================
embedding_dims = {}
for col in cat_cols_list:
    n_categories = X_train[col].nunique()
    embedding_dims[col] = n_categories

embedding_sizes = {}
for col, n_cat in embedding_dims.items():
    emb_size = min(50, n_cat // 2 + 1)
    embedding_sizes[col] = emb_size

# ==================== DEFINE IMPROVED MODEL ====================
class AdultMLP(nn.Module):
    def __init__(self, embedding_dims, embedding_sizes, num_features, hidden_sizes=[64, 32]):
        super().__init__()

        self.embeddings = nn.ModuleList()
        for col in embedding_dims.keys():
            num_embeddings = embedding_dims[col] + 1
            embedding_dim = embedding_sizes[col]
            self.embeddings.append(nn.Embedding(num_embeddings, embedding_dim))

        total_emb_dim = sum(embedding_sizes.values())
        input_size = total_emb_dim + num_features

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.5),  # INCREASED from 0.3
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.5),  # INCREASED from 0.3
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, X_cat, X_num):
        embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_concat = torch.cat(embedded, dim=1)
        x = torch.cat([embedded_concat, X_num], dim=1)
        return self.fc_layers(x)

# ==================== TRAINING ====================
print("Training improved MLP...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdultMLP(embedding_dims, embedding_sizes, len(num_cols_list))
model.to(device)

n_neg = (y_train_encoded == 0).sum()
n_pos = (y_train_encoded == 1).sum()
pos_weight = n_neg / n_pos

# ADD weight_decay for L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for X_cat, X_num, y in loader:
        X_cat, X_num, y = X_cat.to(device), X_num.to(device), y.to(device)
        logits = model(X_cat, X_num).view(-1)
        loss = loss_fn(logits, y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_cat, X_num, y in loader:
            X_cat, X_num, y = X_cat.to(device), X_num.to(device), y.to(device)
            logits = model(X_cat, X_num).view(-1)
            loss = loss_fn(logits, y.view(-1))
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y.view(-1)).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total

# Early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 5

for epoch in range(50):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'app/models/mlp_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(torch.load('app/models/mlp_model.pth'))
            break

# ==================== EVALUATE ====================
test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Save embedding configuration
joblib.dump(embedding_dims, 'app/models/embedding_dims.pkl')
joblib.dump(embedding_sizes, 'app/models/embedding_sizes.pkl')

print("\n✅ Model saved to app/models/mlp_model.pth")
