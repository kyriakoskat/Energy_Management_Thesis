import pandas as pd
import pytest
from energy_community.data import Dataset, generate_demo


def test_small_gap_repair_and_conservative_day_exclusion(tmp_path):
    generate_demo(tmp_path, [1, 2], 3)
    path = tmp_path / "H1_15min.csv"
    frame = pd.read_csv(path)
    # Two missing interior intervals may be repaired. A five-interval gap may not.
    frame = frame.drop([20, 21, 96 + 20, 96 + 21, 96 + 22, 96 + 23, 96 + 24])
    frame.to_csv(path, index=False)
    strict = Dataset(tmp_path, [1, 2], repair=False)
    repaired = Dataset(tmp_path, [1, 2], repair=True)
    assert strict.days == ["2020-06-03"]
    assert repaired.days == ["2020-06-01", "2020-06-03"]
    assert repaired.frames[1]["2020-06-01"].consumption_kWh.notna().all()
    assert len(repaired.frames[1]["2020-06-01"]) == 96


def test_duplicate_periods_exclude_the_day(tmp_path):
    generate_demo(tmp_path, [1, 2], 3)
    path = tmp_path / "H1_15min.csv"
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    data = Dataset(tmp_path, [1, 2])
    assert "2020-06-01" not in data.days
    assert any(row["reason"] == "duplicate_periods" for row in data.audit)


def test_sell_above_buy_is_rejected(tmp_path):
    generate_demo(tmp_path, [1, 2], 3)
    path = tmp_path / "ireland_prices_15min.csv"
    frame = pd.read_csv(path)
    frame["sell_price_eur_kwh"] = frame.buy_price_eur_kwh + 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="No complete common days"):
        Dataset(tmp_path, [1, 2])
