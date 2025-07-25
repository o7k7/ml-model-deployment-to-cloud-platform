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
    if "DYNO" in os.environ and os.path.isdir(".dvc"):
        aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        print(f"DEBUG: AWS_ACCESS_KEY_ID is set: {bool(aws_key_id)}")
        print(f"DEBUG: AWS_SECRET_ACCESS_KEY is set: {bool(aws_secret_key)}")

        if not all([aws_key_id, aws_secret_key]):
            print("AWS credentials not found in environment variables")
        else:
            try:
                print("Attempting to run 'dvc pull'...")

                config_command = ["dvc", "config", "core.no_scm", "true"]
                subprocess.run(config_command, check=True)
                print("Set 'core.no_scm=true' in DVC config successfully.")

                dvc_pull_result = subprocess.run(
                    ["dvc", "pull", "--force"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("DVC pull successful.")
                print("DVC Output:", dvc_pull_result.stdout)
            except subprocess.CalledProcessError as e:
                print("--- DVC PULL FAILED ---")
                print(f"Exit Code: {e.returncode}")
                print(f"STDOUT from DVC: {e.stdout}")
                print(f"STDERR from DVC: {e.stderr}")

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
