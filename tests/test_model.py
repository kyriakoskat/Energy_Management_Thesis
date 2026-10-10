import numpy as np
import pytest
from dataclasses import replace
from energy_community.config import BatteryConfig
from energy_community.model import positive_marginal_allocation, exact_shapley


def test_energy_balance_soc_and_modes(model):
    cfg = model.battery
    result = model.solve(model.players, with_schedule=True)
    for i, rows in result.schedule.groupby("player_id"):
        load = model.houses[i].consumption_kWh.to_numpy() / 0.25
        pv = model.houses[i].production_kWh.to_numpy() / 0.25
        residual = (
            rows.grid_import_kw
            + rows.discharge_kw
            + pv
            + rows.internal_in_kw
            - load
            - rows.charge_kw
            - rows.grid_export_kw
            - rows.internal_out_kw
        )
        assert np.max(np.abs(residual)) < 1e-7
        change = rows.soc_end_kwh - rows.soc_start_kwh
        expected = np.sqrt(
            cfg.roundtrip_efficiency
        ) * rows.charge_kw * 0.25 - rows.discharge_kw * 0.25 / np.sqrt(
            cfg.roundtrip_efficiency
        )
        assert np.max(np.abs(change - expected)) < 1e-7
        assert rows.soc_start_kwh.min() >= -1e-8
        assert (
            rows.soc_end_kwh.max() <= cfg.capacity_kwh * cfg.depth_of_discharge + 1e-8
        )
        assert rows.soc_start_kwh.iloc[0] == pytest.approx(
            rows.soc_end_kwh.iloc[-1], abs=1e-7
        )
        assert np.min(np.maximum(rows.charge_kw, rows.discharge_kw)) >= 0
        assert np.max(np.minimum(rows.charge_kw, rows.discharge_kw)) < 1e-7
    trades = result.schedule.groupby("period")[
        ["internal_in_kw", "internal_out_kw"]
    ].sum()
    assert np.max(np.abs(trades.internal_in_kw - trades.internal_out_kw)) < 1e-7
    assert result.cost == pytest.approx(
        result.energy_cost + result.degradation_cost + result.trade_penalty, abs=1e-7
    )


def test_zero_battery_solo_matches_passive_baseline(dataset):
    model = dataset.day_model(
        dataset.days[0], replace(BatteryConfig(), capacity_kwh=0, power_kw=0)
    )
    for i in model.players:
        assert model.solve([i]).cost == pytest.approx(model.baseline([i]), abs=1e-7)


def test_allocation_budget_and_individual_rationality(model):
    allocation = positive_marginal_allocation(model, model.players)
    assert allocation.allocated_bonus.min() >= -1e-8
    assert (allocation.allocated_cost <= allocation.solo_cost + 1e-8).all()
    assert allocation.allocated_cost.sum() == pytest.approx(
        model.solve(model.players).cost, abs=1e-6
    )
    assert allocation.allocated_bonus.sum() == pytest.approx(
        model.value(model.players), abs=1e-6
    )
    shapley = exact_shapley(model, model.players)
    assert sum(shapley.values()) == pytest.approx(model.value(model.players), abs=1e-6)


def test_singleton_and_empty_game(model):
    assert model.value([]) == 0
    assert model.value([1]) == pytest.approx(0, abs=1e-8)
    assert exact_shapley(model, [1])[1] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("roundtrip_efficiency", 0),
        ("depth_of_discharge", 1.5),
        ("capacity_kwh", -1),
        ("trade_epsilon", float("nan")),
    ],
)
def test_invalid_physical_settings(field, value):
    with pytest.raises(ValueError):
        BatteryConfig(**{field: value})
