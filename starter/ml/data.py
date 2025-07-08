import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.ml.model import compute_model_metrics


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def log_performance_on_slices(
        data: pd.DataFrame,
        feature_name: str,
        model,
        encoder,
        lb,
        categorical_features: list,
        label: str,
        output_filename: str
):
    """
    Computes and prints performance metrics for slices of data based on a feature.

    Args:
        data (pd.DataFrame): The dataset to slice.
        feature_name (str): The name of the categorical feature to slice by.
        model: The trained machine learning model.
        encoder: OneHotEncoder
        lb: LabelBinarizer
        categorical_features (list): Full list of categorical features for processing.
        label (str): The name of the target variable column.
    """
    print(f"\n--- Performance on slices for feature: '{feature_name}' ---")
    with open(output_filename, 'a') as f:
        for cls in data[feature_name].unique():
            # Create a slice of the data for the current value
            df_slice = data[data[feature_name] == cls]

            X_slice, y_slice, _, _ = process_data(
                df_slice,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds_slice = model.predict(X_slice)

            precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

            result_line = (f"{feature_name} - Slice '{cls}'"
                           f": Precision={precision:.4f}, "
                           f"Recall={recall:.4f}, F1-Score={fbeta:.4f}\n")
            print(result_line.strip())
            f.write(result_line)

            print(f"\nSlice performance analysis results saved to '{output_filename}'.")
