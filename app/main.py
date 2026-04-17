from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictionInput, PredictionOutput
from app.preprocessing import predict_xgb, predict_mlp
import os

app = FastAPI(title="Adult Income Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Servera frontend"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static/index.html"))

@app.get("/valid-countries")
def get_valid_countries():
    """Returnera giltiga länder från label_encoders"""
    from app.preprocessing import label_encoders
    import pandas as pd
    # Filtrera bort NaN-värden
    countries = [c for c in label_encoders['native-country'].classes_ if pd.notna(c)]
    return {"countries": sorted(countries)}

@app.get("/education-levels")
def get_education_levels():
    """Returnera utbildningsnivåer med namn och mappning"""
    education_mapping = {
        1: "Preschool",
        2: "1st-4th",
        3: "5th-6th",
        4: "7th-8th",
        5: "9th",
        6: "10th",
        7: "11th",
        8: "12th",
        9: "Some-college",
        10: "Assoc-voc",
        11: "Assoc-acdm",
        12: "Bachelors",
        13: "Masters",
        14: "Prof-school",
        15: "Doctorate"
    }
    return {"education_levels": education_mapping}

@app.get("/valid-categories")
def get_valid_categories():
    """Returnera alla giltiga kategorier för categorical features"""
    from app.preprocessing import label_encoders
    import pandas as pd

    categories = {}
    for col in ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
        if col in label_encoders:
            categories[col] = [c for c in label_encoders[col].classes_ if pd.notna(c)]

    return {"categories": categories}

@app.post("/predict/xgboost", response_model=PredictionOutput)
def predict_xgboost(input_data: PredictionInput):
    data_dict = input_data.dict()
    pred, prob = predict_xgb(data_dict)
    return PredictionOutput(prediction=pred, probability=prob, model="xgboost")

@app.post("/predict/mlp", response_model=PredictionOutput)
def predict_mlp_endpoint(input_data: PredictionInput):
    data_dict = input_data.dict()
    pred, prob = predict_mlp(data_dict)
    return PredictionOutput(prediction=pred, probability=prob, model="mlp")

@app.get("/health")
def health():
    return {"status": "ok"}