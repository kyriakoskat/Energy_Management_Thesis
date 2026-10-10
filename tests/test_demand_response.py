import pytest
from energy_community import demand_response as dr


@pytest.mark.parametrize("strategy,parameter", [("cap", 0.8), ("fractional", 0.2)])
def test_shift_conserves_energy_and_benchmark_monotonicity(
    dataset, strategy, parameter
):
    day = dataset.days[0]
    houses = {i: dataset.frames[i][day] for i in dataset.players}
    aggregate = dataset.frames["prices"][day].copy()
    aggregate["consumption_kWh"] = sum(
        f.consumption_kWh.to_numpy() for f in houses.values()
    )
    aggregate["production_kWh"] = sum(
        f.production_kWh.to_numpy() for f in houses.values()
    )
    aggregate["net_load_kWh"] = aggregate.consumption_kWh - aggregate.production_kWh
    baseline = dr.solve_ideal_battery_day(aggregate, 0)
    battery = dr.solve_ideal_battery_day(aggregate, 10)
    assert battery["cost_eur"] <= baseline["cost_eur"] + 1e-7
    activation = dr.select_activation_window(aggregate, 4, "max_import_cost")
    recovery = dr.select_recovery_window(
        aggregate, battery["schedule"], 10, activation, 16, "battery_aware", True
    )
    assert not dr.windows_overlap(
        activation["activation_start_period"],
        activation["activation_end_period"],
        recovery["recovery_start_period"],
        recovery["recovery_end_period"],
    )
    fn = (
        dr.apply_cap_load_shift if strategy == "cap" else dr.apply_fractional_load_shift
    )
    modified, profile, players = fn(aggregate, houses, activation, recovery, parameter)
    assert modified.consumption_kWh.sum() == pytest.approx(
        aggregate.consumption_kWh.sum(), abs=1e-7
    )
    assert (modified.consumption_kWh >= 0).all()
    assert profile.removed_kWh.sum() == pytest.approx(profile.added_kWh.sum(), abs=1e-7)
    assert players.shifted_kwh.sum() == pytest.approx(
        profile.removed_kWh.sum(), abs=1e-7
    )
