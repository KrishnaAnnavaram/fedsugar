import pytest

from fedsugar.data import split, validate
from fedsugar.synthetic import generate


@pytest.fixture(scope="session")
def frame():
    return validate(generate(6000, seed=11))


@pytest.fixture(scope="session")
def data(frame):
    return split(frame, seed=11)
