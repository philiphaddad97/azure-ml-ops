import json
import joblib

import numpy as np
from schema import Schema, Use, Or

from azureml.core.model import Model


def init():
    global svm, xgboost, rf, input_schema
    # Deserialize the model files back into scikit-learn models.
    svm = joblib.load(Model.get_model_path("svm"))
    xgboost = joblib.load(Model.get_model_path("xgboost"))
    rf = joblib.load(Model.get_model_path("rf"))

    # For input validation!
    input_schema = Schema(
        {
            "Model": Or("SVM", "XGBoost", "RF"),
            "HT": {"Mean": Use(float), "STD": Use(float)},
            "PPT": {"Mean": Use(float), "STD": Use(float)},
            "RRT": {"Mean": Use(float), "STD": Use(float)},
            "RPT": {"Mean": Use(float), "STD": Use(float)},
        }
    )


def run(raw_data):
    data = json.loads(raw_data)
    if input_schema.is_valid(data) is False:
        return {"message": "failed"}
    try:
        if data["Model"] == "RF":
            model = rf
        elif data["Model"] == "XGBoost":
            model = xgboost
        else:
            model = svm
        # The Correct order for the features:
        # HT_mean, HT_std, PPT_mean, PPT_std, PRT_mean, RPT_std, RRT_mean, RRT_std
        sample = np.array(
            [
                data["HT"]["Mean"],
                data["HT"]["STD"],
                data["PPT"]["Mean"],
                data["PPT"]["STD"],
                data["RPT"]["Mean"],
                data["RPT"]["STD"],
                data["HT"]["Mean"],
                data["HT"]["STD"],
            ]
        ).reshape(
            1, 8
        )  # 1 sample & 8 features
        result = model.predict(sample)
        return {
            "message": "success",
            "user_id": json.dumps(result.tolist()[0]),
            "used_model": model.__class__.__name__,
        }
    except Exception as e:
        return {"message": str(e)}
