from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 데이터 로드
data = load_wine()
X = data.data
y = data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 훈련
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 모델 저장
with open("wine_model.pkl", "wb") as f:
    pickle.dump(model, f)




# ###########

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# 모델 로드
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

class WineFeatures(BaseModel):
    features: list

@app.post("/predict/")
def predict_wine_quality(wine: WineFeatures):
    try:
        prediction = model.predict([wine.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))