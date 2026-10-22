import requests
import json

url = "http://127.0.0.1:5000/api/generate"
payload = {
    "text": ("This is a typical page of text. " * 80),
    "font_paths": [],  # Should fall back to bundled fonts
    "config": {
        "font_size": 42,
        "line_height": 60,
    }
}

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Session ID: {data.get('session_id')}")
        print(f"Page Count: {data.get('page_count')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
