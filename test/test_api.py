from fastapi.testclient import TestClient
from fastapi import status
import pytest

from polysoleval import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Hello World"}


def test_read_models():
    response = client.get("/models/")
    assert response.status_code == status.HTTP_200_OK
    types = response.json()
    assert len(types) == 2
    assert "Inception3" in types
    assert "Vgg13" in types


@pytest.mark.parametrize("model_type", ["Inception3", "Vgg13"])
def test_read_model_names(model_type: str):
    response = client.get(f"models/{model_type}/")
    assert response.status_code == status.HTTP_200_OK
    names = response.json()
    assert len(names) == 1
    assert names[0] == "AridAgar"
