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
embedding_dims = joblib.load(os.path.join(MODEL_DIR, 'embedding_dims.pkl'))
embedding_sizes = joblib.load(os.path.join(MODEL_DIR, 'embedding_sizes.pkl'))

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

    # Konvertera och lagra cat_features
    cat_features = {}
    for label_col, api_col in cat_cols_mapping.items():
        value = data_dict.get(api_col, '')
        try:
            cat_features[label_col] = label_encoders[label_col].transform([value])[0]
        except ValueError as e:
            # If value not in training data, use first valid class as default
            valid_classes = [c for c in label_encoders[label_col].classes_ if str(c) != 'nan']
            if valid_classes:
                cat_features[label_col] = label_encoders[label_col].transform([valid_classes[0]])[0]
            else:
                raise ValueError(f"No valid classes for {label_col}") from e
    
    # Numeriska features
    num_features = np.array([
        data_dict.get('age', 0),
        data_dict.get('fnlwgt', 0),
        data_dict.get('education_num', 0),
        np.log1p(data_dict.get('capital_gain', 0)),
        np.log1p(data_dict.get('capital_loss', 0)),
        data_dict.get('hours_per_week', 0)
    ], dtype=np.float32).reshape(1, -1)
    
    # Normalisera numeriska features
    num_features_scaled = scaler.transform(num_features)
    
    # Konvertera cat till numeriska
    cat_cols_list = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    cat_features_arr = np.array([cat_features[col] for col in cat_cols_list], dtype=np.int64).reshape(1, -1)
    
    return cat_features_arr, num_features_scaled

def predict_xgb(data_dict):
    """Predict med XGBoost"""
    cat_features_arr, num_features_scaled = preprocess_input(data_dict)

    # Kombinera features
    all_features = np.hstack([cat_features_arr, num_features_scaled])

    # Predictera
    pred_prob = xgb_model.predict_proba(all_features)[0][1]
    prediction = 1 if pred_prob > 0.5 else 0
    return prediction, float(pred_prob)

def predict_mlp(data_dict):
    """Predict med MLP"""
    cat_features_arr, num_features_scaled = preprocess_input(data_dict)

    # Konvertera till torch tensors
    X_cat = torch.tensor(cat_features_arr, dtype=torch.long)
    X_num = torch.tensor(num_features_scaled, dtype=torch.float32)

    # Predictera
    with torch.no_grad():
        logits = mlp_model(X_cat, X_num)
        pred_prob = torch.sigmoid(logits).item()

    prediction = 1 if pred_prob > 0.5 else 0
    return prediction, float(pred_prob)
