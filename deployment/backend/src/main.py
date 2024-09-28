from typing import Annotated
from fastapi import FastAPI, Body
from pydantic import BaseModel
import pickle
from src.text_process import TextPreprocessor
from fastembed import TextEmbedding


app = FastAPI()

preprocessor = TextPreprocessor()
with open("/model/model_transformer.pkl", mode="rb") as file:
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
        predictions = model.predict_proba([preprocessor.preprocess_text(cleaned_text)])[
            0
        ]
    else:
        embedding = next(embeddings.embed(documents=cleaned_text))
        predictions = model.predict_proba([embedding])[0]

    responce = {
        "comment": predict_body.input,
        "predictions": list(map(float, predictions)),
    }

    return responce
