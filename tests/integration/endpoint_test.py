import requests
import json

sample_svm = """{
    "Model": "SVM",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
    },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
}"""
sample_xgboost = """{
    "Model": "XGBoost",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
    },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
}"""
sample_rf = """{
    "Model": "RF",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
    },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
}"""

sample_invalid = """{
    "Model": "R F",
    "HT": {
        "Mean": 48.43,
        "STD": 23.34
    },
    "PPT": {
        "Mean": 120.43,
        "STD": 37.41
    },
    "RRT": {
        "Mean": 124.43,
        "STD": 45.34
    },
    "RPT": {
        "Mean": 132.56,
        "STD": 47.12
    }
}"""


def test_ml_service_svm(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Authorization": ("Bearer " + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(sample_svm), headers=headers)
    assert "success" in resp.json()["message"]
    assert isinstance(int(resp.json()["user_id"]), int)
    assert resp.json()["used_model"] == "SVC"
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get("content-type") == "application/json"
    assert int(resp.headers.get("Content-Length")) > 0


def test_ml_service_rf(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Authorization": ("Bearer " + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(sample_rf), headers=headers)
    assert "success" in resp.json()["message"]
    assert isinstance(int(resp.json()["user_id"]), int)
    assert resp.json()["used_model"] == "RandomForestClassifier"
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get("content-type") == "application/json"
    assert int(resp.headers.get("Content-Length")) > 0


def test_ml_service_xgboost(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Authorization": ("Bearer " + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(sample_xgboost), headers=headers)
    assert resp.json()["message"] == "success"
    assert isinstance(int(resp.json()["user_id"]), int)
    assert resp.json()["used_model"] == "XGBClassifier"
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get("content-type") == "application/json"
    assert int(resp.headers.get("Content-Length")) > 0


def test_ml_service_invalid(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {"Content-Type": "application/json"}
    else:
        headers = {"Authorization": ("Bearer " + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(sample_invalid), headers=headers)
    assert resp.json()["message"] == "failed"
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get("content-type") == "application/json"
    assert int(resp.headers.get("Content-Length")) > 0
