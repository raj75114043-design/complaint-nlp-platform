import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "none")
os.environ.setdefault("API_KEYS", "test-key")


@pytest.fixture(scope="session")
def rows():
    from cnlp.pipeline import load_rows
    return load_rows()


@pytest.fixture(scope="session")
def model(rows):
    from cnlp.model import train
    return train(rows, "test")
