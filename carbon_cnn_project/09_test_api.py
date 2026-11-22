import json
import time

import requests


API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30


def test_api():
    print("Checking API health...")
    health_response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
    health_response.raise_for_status()
    health = health_response.json()
    if not health.get("model_loaded"):
        raise ValueError("The API is running but has no model loaded.")
    print(f"Model loaded: {health['model_name']}")

    scenarios = {
        "Normal sensor readings": {
            "temperature": 25.5,
            "humidity": 60.2,
            "wind_speed": 12.0,
            "industry_level": 2,
            "vehicle_count": 5000,
            "energy_usage": 150.5,
        },
        "Extreme sensor readings": {
            "temperature": 48.0,
            "humidity": 90.0,
            "wind_speed": 2.0,
            "industry_level": 5,
            "vehicle_count": 80000,
            "energy_usage": 900.0,
        },
    }
    for name, payload in scenarios.items():
        print(f"\n{name}")
        start = time.perf_counter()
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000
        prediction = response.json()
        for field in ("predicted_emission", "status_level", "confidence"):
            if field not in prediction:
                raise ValueError(f"Prediction response is missing {field!r}.")
        print(json.dumps(prediction, indent=2))
        print(f"Request time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    try:
        test_api()
    except requests.ConnectionError:
        raise SystemExit(
            "Cannot reach the API. From the repository root, run "
            "python -m uvicorn carbon_cnn_project.09_fastapi_server:app --reload"
        )
    except (requests.RequestException, ValueError) as error:
        raise SystemExit(f"API smoke test failed: {error}")
