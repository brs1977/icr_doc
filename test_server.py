import pytest
import os
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_read_main():        
    filename = 'data/ScanImage388.jpg'
    print(filename)
    with open(filename, mode='rb') as test_file:
        files = {"file": (os.path.basename(filename), test_file, "image/jpeg")}
        response = client.post("/predict", files=files)

        print(response.text)

    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/json'
