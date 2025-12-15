import pytest
from fastapi.testclient import TestClient
import numpy as np

# This import assumes the test file is run from the root of the project
# and 'src' is in the python path.
# You might need to adjust the path depending on your test runner configuration.
# Example: PYTHONPATH=. pytest
from src.hyper_adaptive_assimilator import app, API_KEY

client = TestClient(app)

USER_ID = "test_user_001"
NEW_USER_ID = "new_test_user_002"
TASK_ID = 0
HEADERS = {"X-API-KEY": API_KEY}

# Sample data for testing, matching INPUT_DIM=3
TRAIN_DATA = [[np.sin(i), np.cos(i), i**2 / 16] for i in np.linspace(-4, 4, 100)]
PREDICT_DATA = [[np.sin(0.5), np.cos(0.5), 0.5**2 / 16]]

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """Ensures a clean state by removing test model files before and after tests."""
    import os
    from src.hyper_adaptive_assimilator import get_model_path

    # --- Setup: Clean before tests ---
    print("\n--- Cleaning up pre-existing test files ---")
    user_model_path = get_model_path(USER_ID)
    if os.path.exists(user_model_path):
        os.remove(user_model_path)

    new_user_model_path = get_model_path(NEW_USER_ID)
    if os.path.exists(new_user_model_path):
        os.remove(new_user_model_path)

    yield # Let the tests run

    # --- Teardown: Clean after tests ---
    print("\n--- Cleaning up test files post-execution ---")
    if os.path.exists(user_model_path):
        os.remove(user_model_path)

    if os.path.exists(new_user_model_path):
        os.remove(new_user_model_path)

def test_root_is_not_found():
    response = client.get("/")
    assert response.status_code == 404

def test_assimilation_and_prediction_cycle():
    # 1. Assimilate (Train) the model - now a background task
    assimilate_payload = {
        "task_id": TASK_ID,
        "data": TRAIN_DATA
    }
    response_assimilate = client.post(f"/assimilate/{USER_ID}", json=assimilate_payload, headers=HEADERS)

    assert response_assimilate.status_code == 200
    assert response_assimilate.json() == {
        "status": "assimilation_queued",
        "user_id": USER_ID,
        "task_id": TASK_ID
    }

    # Allow time for the background task to complete.
    # In a real-world scenario, you would use a more robust mechanism
    # like a webhook, a status polling endpoint, or a message queue.
    import time
    print("\nWaiting for background training to complete...")
    time.sleep(30) # A simple but effective way for this test case.

    # 2. Predict using the trained model
    predict_payload = {
        "task_id": TASK_ID,
        "data": PREDICT_DATA
    }
    response_predict = client.post(f"/predict/{USER_ID}", json=predict_payload, headers=HEADERS)

    assert response_predict.status_code == 200
    prediction = response_predict.json()

    # Verification
    assert isinstance(prediction, list)
    assert len(prediction) == 1
    assert len(prediction[0]) == 3 # Output dimension should match input dimension

    for val in prediction[0]:
        assert isinstance(val, float)

def test_predict_without_assimilation():
    # This test will create a new, untrained model for a new user.
    predict_payload = {
        "task_id": 0,
        "data": PREDICT_DATA
    }
    response_predict = client.post(f"/predict/{NEW_USER_ID}", json=predict_payload, headers=HEADERS)

    assert response_predict.status_code == 200
    prediction = response_predict.json()
    assert len(prediction) == 1
    assert len(prediction[0]) == 3

def test_invalid_api_key():
    wrong_headers = {"X-API-KEY": "wrong_key"}
    assimilate_payload = {"task_id": 0, "data": []}
    response = client.post(f"/assimilate/{USER_ID}", json=assimilate_payload, headers=wrong_headers)
    assert response.status_code == 403

    predict_payload = {"task_id": 0, "data": []}
    response = client.post(f"/predict/{USER_ID}", json=predict_payload, headers=wrong_headers)
    assert response.status_code == 403

def test_invalid_data_dimensions():
    # Assimilate with wrong dimensions
    invalid_data = [[0.1, 0.2], [0.3, 0.4]] # 2 features instead of 3
    assimilate_payload = {"task_id": 0, "data": invalid_data}
    response = client.post(f"/assimilate/{USER_ID}", json=assimilate_payload, headers=HEADERS)
    assert response.status_code == 400

    # Predict with wrong dimensions
    predict_payload = {"task_id": 0, "data": invalid_data}
    response = client.post(f"/predict/{USER_ID}", json=predict_payload, headers=HEADERS)
    assert response.status_code == 400
