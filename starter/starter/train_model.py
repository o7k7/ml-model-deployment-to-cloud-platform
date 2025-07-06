# Script to train machine learning model.
import os

import joblib
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data, log_performance_on_slices
import pandas as pd

from starter.starter.ml.model import train_model, compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def save_artifact(artifact, filename: str):
    artifacts_dir = "../model"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save artifact
    print("Saving artifact..." + " " + filename)
    joblib.dump(artifact, os.path.join(artifacts_dir, filename + '.joblib'))
    print("Artifact saved successfully.")


def load_artifact(filename: str):
    artifacts_dir = "model"
    try:
        print("Loading artifact..." + " " + filename)
        artifact = joblib.load(os.path.join(
            artifacts_dir, filename + ".joblib"))
        print("Artifact loaded successfully.")

        return artifact
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        return None


def read_data_into_df(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    return data


if __name__ == '__main__':
    output_filename = "../model/slice_output.txt"
    if os.path.exists(output_filename):
        os.remove(output_filename)

    y_label = 'salary'
    # Add code to load in the data.
    data = read_data_into_df("../data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[y_label])
    print('Train Data Shape:', train.shape)
    print('Test Data Shape:', test.shape)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=y_label, training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features,
                                        label=y_label, training=False,
                                        encoder=encoder, lb=lb)

    # Train and save a model.
    model = train_model(X_train, y_train)

    save_artifact(model, "random_forest_model")
    save_artifact(encoder, "one_hot_encoder")
    save_artifact(lb, "lb")

    predictions = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    for feature in cat_features:
        log_performance_on_slices(
            data=test,
            feature_name=feature,
            model=model,
            encoder=encoder,
            lb=lb,
            categorical_features=cat_features,
            label="salary",
            output_filename=output_filename
        )

    print("---- GENERAL SCORES ----")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", fbeta)
