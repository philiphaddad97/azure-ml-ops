import requests
import json

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

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(sample_rf), headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0
