"""Validated experiment settings. Paths are relative to the configuration file."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 10.0
    power_kw: float = 3.3
    roundtrip_efficiency: float = 0.95
    depth_of_discharge: float = 0.85
    initial_soc_fraction: float = 0.5
    cyclic_soc: bool = True
    blcoe_eur_kwh: float = 0.05
    degradation_weight: float = 0.5
    grid_import_max_kw: float | None = 20.0
    grid_export_max_kw: float | None = 20.0
    fixed_cost_eur: float = 0.0
    trade_epsilon: float = 0.001
    solver_time_limit_seconds: float | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"battery.{key} must be finite and non-negative")
        if not 0 < self.roundtrip_efficiency <= 1:
            raise ValueError("roundtrip_efficiency must be in (0, 1]")
        if (
            not 0 <= self.depth_of_discharge <= 1
            or not 0 <= self.initial_soc_fraction <= 1
        ):
            raise ValueError("SOC fractions must be in [0, 1]")
        if self.solver_time_limit_seconds == 0:
            raise ValueError("solver_time_limit_seconds must be positive or null")


@dataclass(frozen=True)
class RepresentativeConfig:
    clusters: int = 12
    initializations: int = 50
    max_iterations: int = 100
    standardize: bool = True


@dataclass(frozen=True)
class FormationConfig:
    method: str = "auto"
    group_size: int = 4
    exact_max_players: int = 30
    time_limit_seconds: float = 120.0
    local_starts: int = 100
    local_passes: int = 100
    local_patience: int = 1000
    random_partitions: int = 5


@dataclass(frozen=True)
class DemandResponseConfig:
    enabled: bool = True
    capacities_kwh: list[float] = field(default_factory=lambda: [0.0, 20.0, 80.0])
    reference_capacities_kwh: list[float] = field(default_factory=lambda: [185.0])
    strategy: str = "cap"
    cap_ratios: list[float] = field(default_factory=lambda: [0.8])
    reduction_fractions: list[float] = field(default_factory=lambda: [0.2])
    recovery_modes: list[str] = field(default_factory=lambda: ["battery_aware"])
    activation_periods: int = 4
    recovery_periods: int = 16
    activation_metric: str = "max_import_cost"
    allow_recovery_before_activation: bool = True


@dataclass(frozen=True)
class Config:
    name: str = "experiment"
    players: list[int] = field(default_factory=lambda: list(range(1, 21)))
    data_dir: str = "../data/private/processed_data_clean"
    output_dir: str = "../outputs/thesis_20"
    partition_file: str | None = None
    start_day: str | None = None
    end_day: str | None = None
    seed: int = 42
    demo: bool = False
    demo_days: int = 6
    repair_small_gaps: bool = False
    stability: bool = True
    shapley_players: list[int] = field(default_factory=list)
    shapley_start_day: str | None = None
    shapley_end_day: str | None = None
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    representative: RepresentativeConfig = field(default_factory=RepresentativeConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
    demand_response: DemandResponseConfig = field(default_factory=DemandResponseConfig)

    def validate(self):
        if len(self.players) < 2 or any(
            type(p) is not int or p <= 0 for p in self.players
        ):
            raise ValueError("players must contain at least two positive integer IDs")
        if len(set(self.players)) != len(self.players):
            raise ValueError("players must be unique")
        f, r, dr = self.formation, self.representative, self.demand_response
        if f.group_size < 2 or len(self.players) % f.group_size:
            raise ValueError("players must be divisible by formation.group_size (>= 2)")
        if f.method not in {"auto", "milp", "local_search"}:
            raise ValueError("formation.method must be auto, milp or local_search")
        counts = [
            r.clusters,
            r.initializations,
            r.max_iterations,
            f.local_starts,
            f.local_passes,
            f.local_patience,
        ]
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("clustering and search counts must be positive")
        if f.random_partitions < 0 or f.time_limit_seconds <= 0:
            raise ValueError("invalid formation time limit or random partition count")
        if self.demo and self.demo_days < 3:
            raise ValueError("demo_days must be at least 3")
        if len(self.shapley_players) > 10 or len(set(self.shapley_players)) != len(
            self.shapley_players
        ):
            raise ValueError("exact Shapley requires at most 10 unique players")
        if not set(self.shapley_players).issubset(self.players):
            raise ValueError("shapley_players must be part of players")
        if dr.strategy not in {"cap", "fractional"}:
            raise ValueError("DR strategy must be cap or fractional")
        if not dr.capacities_kwh or not dr.recovery_modes:
            raise ValueError("DR requires capacities and recovery modes")
        if any(
            c < 0 or not math.isfinite(c)
            for c in dr.capacities_kwh + dr.reference_capacities_kwh
        ):
            raise ValueError("DR capacities must be finite and non-negative")
        if any(not 0 <= x <= 1 for x in dr.cap_ratios + dr.reduction_fractions):
            raise ValueError("DR fractions must be in [0, 1]")
        if not set(dr.recovery_modes).issubset(
            {"cheapest_price", "battery_aware", "surplus_first"}
        ):
            raise ValueError("unknown DR recovery mode")
        if dr.activation_metric not in {
            "max_import_cost",
            "max_net_cost",
            "max_net_load",
            "max_consumption",
        }:
            raise ValueError("unknown activation metric")
        if (
            min(dr.activation_periods, dr.recovery_periods) < 1
            or dr.activation_periods + dr.recovery_periods > 96
        ):
            raise ValueError("DR windows must fit within 96 periods")
        import pandas as pd

        for start, end in [
            (self.start_day, self.end_day),
            (self.shapley_start_day, self.shapley_end_day),
        ]:
            if start is not None:
                pd.Timestamp(start)
            if end is not None:
                pd.Timestamp(end)
            if start and end and pd.Timestamp(start) > pd.Timestamp(end):
                raise ValueError("start_day must precede end_day")
        return self


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        for key, cls in [
            ("battery", BatteryConfig),
            ("representative", RepresentativeConfig),
            ("formation", FormationConfig),
            ("demand_response", DemandResponseConfig),
        ]:
            if key in raw:
                raw[key] = cls(**raw[key])
        config = Config(**raw).validate()
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    return config, path.parent
