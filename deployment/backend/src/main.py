import os
import pickle
from typing import Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Body
from fastembed import TextEmbedding
from src.text_process import TextPreprocessor


app = FastAPI()

preprocessor = TextPreprocessor()
with open(f"/model/{os.environ['MODEL_FILENAME']}", mode="rb") as file:
    model = pickle.load(file)

if len(model.steps) != 2:
    embeddings = TextEmbedding(
        model_name="intfloat/multilingual-e5-large", cache_dir="/model"
    )


class PredictBody(BaseModel):
    input: str


@app.post("/predict", tags="Predict")
def predict(predict_body: Annotated[PredictBody, Body()]):
    cleaned_text = preprocessor.clean_text(predict_body.input)

    if len(model.steps) == 2:
        predictions = model.predict_proba([preprocessor.preprocess_text(cleaned_text)])[0]
    else:
        embedding = next(embeddings.embed(documents=cleaned_text))
        predictions = model.predict_proba([embedding])[0]

    responce = {
        "comment": predict_body.input,
        "predictions": list(map(float, predictions)),
    }

    return responce
        