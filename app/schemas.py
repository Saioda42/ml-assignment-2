from pydantic import BaseModel

class PredictionInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictionOutput(BaseModel):
    prediction: int  # 0 eller 1
    probability: float
    model: str  # "mlp" eller "xgboost"