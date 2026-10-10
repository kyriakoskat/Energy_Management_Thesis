import pytest
from energy_community.data import Dataset, generate_demo
from energy_community.config import BatteryConfig


@pytest.fixture
def dataset(tmp_path):
    generate_demo(tmp_path, [1, 2, 3, 4], days=3, seed=42)
    return Dataset(tmp_path, [1, 2, 3, 4])


@pytest.fixture
def model(dataset):
    return dataset.day_model(dataset.days[0], BatteryConfig())
