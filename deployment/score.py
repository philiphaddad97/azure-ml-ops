import json
import joblib

import numpy as np
from schema import Schema, And, Use, Optional, SchemaError, Or

from azureml.core.model import Model


def init():
    global svm, xgboost, rf
    # Deserialize the model files back into scikit-learn (pipeline).
    svm = joblib.load(Model.get_model_path("svm"))
    xgboost = joblib.load(Model.get_model_path("xgboost"))
    rf = joblib.load(Model.get_model_path("rf"))


def run(raw_data):
    input_schema = Schema(
        {
            "Model": Or("SVM", "XGBoost", "RF"),
            "HT": {"Mean": And(Use(float)), "STD": And(Use(float))},
            "PPT": {"Mean": And(Use(float)), "STD": And(Use(float))},
            "RRT": {"Mean": And(Use(float)), "STD": And(Use(float))},
            "RPT": {"Mean": And(Use(float)), "STD": And(Use(float))},
        }
    )
    data = json.loads(raw_data)
    if input_schema.is_valid(data) is False:
        return {"message": "Failed, the input json is not valid!"}
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
        ).reshape(1, 8)
        result = model.predict(sample)
        return {
            "UserID": json.dumps(result.tolist()[0]),
            "Model": model.__class__.__name__,
        }
    except Exception as e:
        return {"message": str(e)}
