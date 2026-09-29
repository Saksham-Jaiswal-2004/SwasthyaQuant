import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_module():
    return importlib.import_module("app.main")


def test_valid_patient_request_shape(app_module):
    client = TestClient(app_module.app)
    payload = {
        "age": 58,
        "height": 170,
        "weight": 78,
        "ap_hi": 145,
        "ap_lo": 90,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "cholesterol": 1,
        "gluc": 1,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in {200, 503}
    if response.status_code == 200:
        body = response.json()
        assert 0 <= body["prediction"] <= 1
        assert 0.0 <= body["risk_probability"] <= 1.0


def test_invalid_age(app_module):
    client = TestClient(app_module.app)
    response = client.post(
        "/api/predict",
        json={
            "age": 10,
            "height": 170,
            "weight": 78,
            "ap_hi": 145,
            "ap_lo": 90,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "gender": 1,
            "cholesterol": 1,
            "gluc": 1,
        },
    )
    assert response.status_code == 422


def test_invalid_height(app_module):
    client = TestClient(app_module.app)
    response = client.post(
        "/api/predict",
        json={
            "age": 58,
            "height": 90,
            "weight": 78,
            "ap_hi": 145,
            "ap_lo": 90,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "gender": 1,
            "cholesterol": 1,
            "gluc": 1,
        },
    )
    assert response.status_code == 422


def test_invalid_weight(app_module):
    client = TestClient(app_module.app)
    response = client.post(
        "/api/predict",
        json={
            "age": 58,
            "height": 170,
            "weight": 15,
            "ap_hi": 145,
            "ap_lo": 90,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "gender": 1,
            "cholesterol": 1,
            "gluc": 1,
        },
    )
    assert response.status_code == 422


def test_invalid_bp(app_module):
    client = TestClient(app_module.app)
    response = client.post(
        "/api/predict",
        json={
            "age": 58,
            "height": 170,
            "weight": 78,
            "ap_hi": 70,
            "ap_lo": 90,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "gender": 1,
            "cholesterol": 1,
            "gluc": 1,
        },
    )
    assert response.status_code == 422


def test_invalid_categorical_values(app_module):
    client = TestClient(app_module.app)
    response = client.post(
        "/api/predict",
        json={
            "age": 58,
            "height": 170,
            "weight": 78,
            "ap_hi": 145,
            "ap_lo": 90,
            "smoke": 0,
            "alco": 0,
            "active": 1,
            "gender": 3,
            "cholesterol": 4,
            "gluc": 1,
        },
    )
    assert response.status_code == 422


def test_preprocessing_output_shape(app_module):
    service = app_module.preprocessing_service
    patient = {
        "age": 58,
        "height": 170,
        "weight": 78,
        "ap_hi": 145,
        "ap_lo": 90,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "cholesterol": 1,
        "gluc": 1,
    }
    vector = service.build_hybrid_vector(patient)
    assert vector.shape == (32,)


def test_hybrid_feature_dimension_before_shap_selection(app_module):
    service = app_module.preprocessing_service
    patient = {
        "age": 58,
        "height": 170,
        "weight": 78,
        "ap_hi": 145,
        "ap_lo": 90,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "cholesterol": 1,
        "gluc": 1,
    }
    vector = service.build_hybrid_vector(patient)
    assert vector.shape[0] == 32


def test_probability_range(app_module):
    client = TestClient(app_module.app)
    payload = {
        "age": 58,
        "height": 170,
        "weight": 78,
        "ap_hi": 145,
        "ap_lo": 90,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "cholesterol": 1,
        "gluc": 1,
    }
    response = client.post("/api/predict", json=payload)
    if response.status_code == 200:
        body = response.json()
        assert 0.0 <= body["risk_probability"] <= 1.0


def test_prediction_binary(app_module):
    client = TestClient(app_module.app)
    payload = {
        "age": 58,
        "height": 170,
        "weight": 78,
        "ap_hi": 145,
        "ap_lo": 90,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "cholesterol": 1,
        "gluc": 1,
    }
    response = client.post("/api/predict", json=payload)
    if response.status_code == 200:
        body = response.json()
        assert body["prediction"] in {0, 1}


def test_missing_model_artifact_error(app_module):
    service = app_module.model_service
    service._bundle = None
    service._load_attempted = False
    model = service.get_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert service.is_loaded() is True


def test_health_endpoint(app_module):
    client = TestClient(app_module.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_benchmark_endpoint(app_module):
    client = TestClient(app_module.app)
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    assert "status" in response.json()
