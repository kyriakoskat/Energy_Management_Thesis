"""Optional comparison using the owner's prepared data and recorded thesis outputs."""

from pathlib import Path
import os
import pandas as pd
import pytest
from energy_community.data import Dataset
from energy_community.config import BatteryConfig

DATA_DIR = os.environ.get("ENERGY_THESIS_DATA_DIR")
pytestmark = pytest.mark.skipif(
    not DATA_DIR, reason="Set ENERGY_THESIS_DATA_DIR to the private prepared CSV folder"
)


def test_recorded_twenty_player_daily_costs():
    reference = Path(__file__).resolve().parents[1] / "docs/reference_results"
    expected = pd.read_csv(reference / "thesis_20_daily_costs.csv").set_index("day")
    partition = [
        list(map(int, m.split(",")))
        for m in pd.read_csv(reference / "thesis_20_subcoalitions.csv").members
    ]
    dataset = Dataset(DATA_DIR, list(range(1, 21)))
    for day in ["2020-01-02", "2020-06-15", "2020-12-11"]:
        model = dataset.day_model(day, BatteryConfig())
        actual = [
            model.baseline(),
            sum(model.solo_costs().values()),
            model.solve(dataset.players).cost,
            sum(model.solve(g).cost for g in partition),
        ]
        row = expected.loc[
            day,
            [
                "baseline_total_cost_eur",
                "solo_total_cost_eur",
                "grand_coalition_cost_eur",
                "subcoalitions_total_cost_eur",
            ],
        ]
        assert actual == pytest.approx(row.tolist(), abs=1e-5)
    assert sum(
        dataset.day_model(d, BatteryConfig()).baseline() for d in dataset.days
    ) == pytest.approx(expected.baseline_total_cost_eur.sum(), abs=1e-6)
