import requests
import json

url = "https://census-ml-bbf664a1cde3.herokuapp.com/infer"


payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}


headers = {
    "Content-Type": "application/json"
}


def send_prediction_request(api_url: str, data: dict, http_headers: dict):
    """
    Sends a POST request to the `infer` endpoint and returns the
    status code and the JSON response body.
    """
    try:
        response = requests.post(api_url, headers=http_headers, data=json.dumps(data))
        status_code = response.status_code

        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Failed to decode response", "content": response.text}

        return status_code, result

    except requests.exceptions.RequestException as e:
        return None, {"error": f"An error occurred: {e}"}


if __name__ == "__main__":
    print(f"Sending POST request to: {url}")

    status, inference_result = send_prediction_request(url, payload, headers)

    print("\n--- Response from API ---")
    if status is not None:
        print(f"Status Code: {status}")
        print("Inference Result (JSON Body):")
        print(json.dumps(inference_result, indent=4))
    else:
        print("Request failed.")
        print(inference_result)
