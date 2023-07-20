# test_api_test.py

import pytest
import requests

def test_index():
    response = requests.get('http://127.0.0.1:80/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Hello, stranger'}

def test_score_min():
    response = requests.get('http://127.0.0.1:80/score_min/')
    assert response.status_code == 200
    assert response.json() == {"score_min" : 0.55}

def test_predict():
    client_id = 100001
    response = requests.get(f'http://127.0.0.1:80/predict?client_id={client_id}')
    assert response.status_code == 200
    assert "proba" in response.json()
    assert isinstance(response.json()["proba"], float)

def test_feats():
    response = requests.get('http://127.0.0.1:80/feats/')
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_importances():
    client_id = 100001
    response = requests.get(f'http://127.0.0.1:80/importances?client_id={client_id}')
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_bar():
    client_id = 100001
    feature = "EXT_SOURCE_3"
    response = requests.get(f'http://127.0.0.1:80/bar?client_id={client_id}&feature={feature}')
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_boxplot():
    feature = "EXT_SOURCE_3"
    response = requests.get(f'http://127.0.0.1:80/boxplot?feature={feature}')
    assert response.status_code == 200
    assert isinstance(response.json(), list)
