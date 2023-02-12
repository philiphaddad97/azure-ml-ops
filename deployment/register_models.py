from azureml.core import Workspace
from azureml.core.model import Model

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws_name", type=str, help="AML workspace name")
    parser.add_argument("--subscription_id", type=str, help="Azure subscription id")
    parser.add_argument("--rg_name", type=str, help="Resource group name")
    args = parser.parse_args()
    print(args)
    ws = Workspace.get(
        name=args["ws_name"],
        subscription_id=args["subscription_id"],
        resource_group=args["rg_name"],
    )
    print(ws)
    # svm = Model.register(model_path="models/svc.pkl", model_name="svm", workspace=ws)
    # xgboost = Model.register(model_path="models/xgb_classifier.pkl", model_name="xgboost", workspace=ws)
    # rf = Model.register(model_path="models/random_forest_classifier.pkl", model_name="rf", workspace=ws)


if __name__ == "__main__":
    main()
