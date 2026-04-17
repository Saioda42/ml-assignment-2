import joblib
import numpy as np
import torch
import torch.nn as nn
import os

# Få rätt väg
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ladda preprocessors
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))

# Definierna AdultMLP samma som notebooken
class AdultMLP(nn.Module):
    def __init__(self, embedding_dims, embedding_sizes, num_features, hidden_sizes=[64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=emb_size + 1, embedding_dim=embedding_sizes[col])
            for col, emb_size in embedding_dims.items()
        ])
        total_emb_dim = sum(embedding_sizes.values())
        input_size = total_emb_dim + num_features
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_sizes[1], 1)
        )
    
    def forward(self, X_cat, X_num):
        embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_concat = torch.cat(embedded, dim=1)
        x = torch.cat([embedded_concat, X_num], dim=1)
        return self.fc_layers(x)

# Ladda MLP
embedding_dims = {'workclass': 10, 'marital_status': 7, 'occupation': 16, 'relationship': 6, 'race': 5, 'sex': 2, 'native_country': 42}
embedding_sizes = {'workclass': 6, 'marital_status': 4, 'occupation': 9, 'relationship': 4, 'race': 3, 'sex': 2, 'native_country': 22}

mlp_model = AdultMLP(embedding_dims, embedding_sizes, num_features=6)
mlp_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mlp_model.pth'), map_location='cpu'))
mlp_model.eval()

def preprocess_input(data_dict):
    """Konvertera input-dict till features"""
    # Mappning från label_encoder-kolumner (bindestreck) till API-kolumner (underscore)
    cat_cols_mapping = {
        'workclass': 'workclass',
        'marital-status': 'marital_status',
        'occupation': 'occupation',
        'relationship': 'relationship',
        'race': 'race',
        'sex': 'sex',
        'native-country': 'native_country'
    }
    num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    # Encode kategoriska
    X_cat = []
    for encoder_col, api_col in cat_cols_mapping.items():
        encoded = label_encoders[encoder_col].transform([data_dict[api_col]])[0]
        X_cat.append(encoded)
    
    # Numeriska
    X_num = [data_dict[col] for col in num_cols]
    
    # Log1p på capital gains/losses
    X_num[3] = np.log1p(X_num[3])  # capital_gain
    X_num[4] = np.log1p(X_num[4])  # capital_loss
    
    # Scale numeriska
    X_num_scaled = scaler.transform([X_num])[0]
    
    return np.array([X_cat]), np.array([X_num_scaled])

def predict_xgb(data_dict):
    X_cat, X_num = preprocess_input(data_dict)
    # XGBoost behöver båda concat
    X_full = np.concatenate([X_cat, X_num], axis=1)
    prob = xgb_model.predict_proba(X_full)[0, 1]
    pred = xgb_model.predict(X_full)[0]
    return int(pred), float(prob)

def predict_mlp(data_dict):
    X_cat, X_num = preprocess_input(data_dict)
    with torch.no_grad():
        X_cat_t = torch.tensor(X_cat, dtype=torch.long)
        X_num_t = torch.tensor(X_num, dtype=torch.float32)
        logit = mlp_model(X_cat_t, X_num_t).squeeze().item()
        prob = 1 / (1 + np.exp(-logit))  # sigmoid
    pred = 1 if prob > 0.5 else 0
    return pred, float(prob)