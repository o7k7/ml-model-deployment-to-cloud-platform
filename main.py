import os
import subprocess
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Body, HTTPException, Request

from starter.dtos.CensusInputRequestDto import CensusInput, body_to_df
from starter.ml.data import process_data
from starter.train_model import cat_features

ml_resources = {}


def load_resources():
    artifacts_dir = "model"

    try:
        print("Loading artifacts...")
        model = joblib.load(os.path.join(artifacts_dir, "random_forest_model.joblib"))
        encoder = joblib.load(os.path.join(artifacts_dir, "one_hot_encoder.joblib"))
        lb = joblib.load(os.path.join(artifacts_dir, "lb.joblib"))
        print("Artifacts loaded successfully.")
        return model, encoder, lb
    except FileNotFoundError:
        print("FATAL ERROR: Model artifacts not found")
        return None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    subprocess.run(["dvc", "pull", "--force"], check=True)

    model, encoder, lb = load_resources()
    if not all([model, encoder, lb]):
        raise Exception("Model artifacts not found")
    # Load the ML model
    app.state.model = model
    app.state.encoder = encoder
    app.state.lb = lb
    yield
    ml_resources.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def health_check():
    return {"status": "Ready"}


@app.post("/infer")
def read_item(
        request: Request,
        input_data: CensusInput = Body(examples=[CensusInput.Example.schema])
):
    input_df = body_to_df(input_data=input_data)

    encoder = request.app.state.encoder
    lb = request.app.state.lb
    model = request.app.state.model

    try:
        processed_input, _, _, _ = process_data(
            X=input_df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")

    try:
        prediction_numerical = model.predict(processed_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    prediction_label = lb.inverse_transform(prediction_numerical)

    return {"predicted_salary": prediction_label[0]}
