from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest

from energy_community.config import (
    Config,
    FormationConfig,
    RepresentativeConfig,
    DemandResponseConfig,
)
from energy_community.experiments import run, Experiment
from energy_community.clustering import run_kmedoids


def small_config():
    return Config(
        name="Integration test",
        demo=True,
        demo_days=3,
        players=[1, 2, 3, 4],
        formation=FormationConfig(
            group_size=2, time_limit_seconds=10, random_partitions=1
        ),
        representative=RepresentativeConfig(clusters=1, initializations=2),
        shapley_players=[1, 2, 3, 4],
        demand_response=DemandResponseConfig(
            capacities_kwh=[0, 10], reference_capacities_kwh=[20]
        ),
    )


def test_whole_pipeline_and_safe_resume(tmp_path, monkeypatch):
    config = small_config()
    output = run(config, tmp_path, output=tmp_path / "result")
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["completed_stages"]) == {
        "prepare",
        "representative",
        "scores",
        "formation",
        "evaluate",
        "allocation",
        "shapley",
        "stability",
        "demand_response",
        "report",
    }
    reps = pd.read_csv(output / "representative_days.csv")
    assert reps.weight.sum() == 3
    assert not reps.day.duplicated().any()
    allocation = pd.read_csv(output / "daily_allocations.csv")
    costs = pd.read_csv(output / "daily_subcoalition_costs.csv").set_index(
        ["day", "coalition_id"]
    )
    allocated = allocation.groupby(["day", "coalition_id"]).allocated_cost.sum()
    assert np.allclose(allocated.sort_index(), costs.cost_eur.sort_index(), atol=1e-6)
    assert (output / "report.md").is_file()
    assert (output / "plots/cumulative_cost.png").stat().st_size > 1000
    comparison = pd.read_csv(output / "comparison_to_full_battery.csv")
    assert np.allclose(
        comparison.battery_only_saving_eur + comparison.saving_eur,
        comparison.combined_saving_eur,
    )
    cached = {p.name: p.read_bytes() for p in (output / "cache").glob("*.json")}
    run(config, tmp_path, output=output)
    assert cached == {p.name: p.read_bytes() for p in (output / "cache").glob("*.json")}
    with pytest.raises(ValueError, match="different settings or inputs"):
        Experiment(replace(config, seed=43), tmp_path, output)
    original = Experiment.allocation

    def interrupted(self):
        raise RuntimeError("Test interruption")

    monkeypatch.setattr(Experiment, "allocation", interrupted)
    with pytest.raises(RuntimeError, match="Test interruption"):
        run(config, tmp_path, stage="allocation", output=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert not {"allocation", "stability", "report"} & set(manifest["completed_stages"])
    monkeypatch.setattr(Experiment, "allocation", original)
    run(config, tmp_path, output=output)
    assert (
        "report"
        in json.loads((output / "manifest.json").read_text())["completed_stages"]
    )


def test_tied_profiles_have_distinct_medoids_and_nonempty_clusters():
    best, _ = run_kmedoids(np.zeros((5, 5)), 3, n_init=2)
    assert len(set(best["medoids"])) == 3
    assert len(set(best["labels"])) == 3


def test_supplied_partition_and_single_day_evaluation(tmp_path):
    config = replace(
        small_config(),
        partition_file="partition.csv",
        start_day="2020-06-01",
        end_day="2020-06-01",
    )
    pd.DataFrame(
        {"coalition_id": [1, 2], "members": ["1,3", "2,4"], "coalition_size": [2, 2]}
    ).to_csv(tmp_path / "partition.csv", index=False)
    output = run(config, tmp_path, stage="report", output=tmp_path / "result")
    assert (
        json.loads((output / "formation_summary.json").read_text())["method"]
        == "supplied_partition"
    )
    assert len(pd.read_csv(output / "daily_evaluation.csv")) == 1
    assert (output / "report.md").exists()
