"""Common scheduling MILP, preserving the original thesis objective and constraints.

Household inputs use kWh per interval. Model variables use kW; DT is 0.25 h.
Individual scheduling excludes simultaneous import/export. Cooperative scheduling
uses the original directed internal trades and their small regularization penalty.
"""

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
import time

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from .config import BatteryConfig

DT = 0.25
T = 96


@dataclass
class SolveResult:
    cost: float
    energy_cost: float
    degradation_cost: float
    trade_penalty: float
    solve_seconds: float
    schedule: pd.DataFrame


class DayModel:
    """A day's data and a coalition-cost cache shared by all experiment stages."""

    def __init__(self, day, houses, prices, battery=None):
        self.day = str(day)
        self.houses = houses
        self.prices = prices
        self.battery = battery or BatteryConfig()
        self.players = tuple(sorted(houses))
        self.cache = {}
        self._solo = None

    def baseline(self, players=None):
        cfg = self.battery
        buy = self.prices.buy_price_eur_kwh.to_numpy(float)
        sell = self.prices.sell_price_eur_kwh.to_numpy(float)
        return float(
            sum(
                np.dot(buy, np.maximum(self.houses[i].net_load_kWh.to_numpy(float), 0))
                - np.dot(
                    sell, np.maximum(-self.houses[i].net_load_kWh.to_numpy(float), 0)
                )
                + cfg.fixed_cost_eur
                for i in (self.players if players is None else players)
            )
        )

    def solo_costs(self):
        if self._solo is None:
            self._solo = {i: self.solve([i]).cost for i in self.players}
        return self._solo

    def value(self, players):
        key = tuple(sorted(players))
        return sum(self.solo_costs()[i] for i in key) - (
            self.solve(key).cost if key else 0.0
        )

    def solve(self, players, with_schedule=False):
        key = tuple(sorted(players))
        if len(key) != len(set(key)) or not set(key).issubset(self.players):
            raise ValueError("coalition must be a unique subset of the day's players")
        if not key:
            return SolveResult(0, 0, 0, 0, 0, pd.DataFrame())
        if key in self.cache and (
            not with_schedule or not self.cache[key].schedule.empty
        ):
            return self.cache[key]
        cfg = self.battery
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            raise RuntimeError("SCIP is unavailable in this OR-Tools installation")
        solver.SetNumThreads(1)
        if cfg.solver_time_limit_seconds is not None:
            solver.SetTimeLimit(round(cfg.solver_time_limit_seconds * 1000))
        inf = solver.infinity()
        usable = cfg.capacity_kwh * cfg.depth_of_discharge
        soc0 = cfg.initial_soc_fraction * usable
        eta = sqrt(cfg.roundtrip_efficiency)
        buy = self.prices.buy_price_eur_kwh.to_numpy(float)
        sell = self.prices.sell_price_eur_kwh.to_numpy(float)
        charge, discharge, imports, exports, soc, mode, trades = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for i in key:
            for t in range(T):
                charge[i, t] = solver.NumVar(0, cfg.power_kw, f"charge_{i}_{t}")
                discharge[i, t] = solver.NumVar(0, cfg.power_kw, f"discharge_{i}_{t}")
                imports[i, t] = solver.NumVar(
                    0,
                    cfg.grid_import_max_kw
                    if cfg.grid_import_max_kw is not None
                    else inf,
                    f"import_{i}_{t}",
                )
                exports[i, t] = solver.NumVar(
                    0,
                    cfg.grid_export_max_kw
                    if cfg.grid_export_max_kw is not None
                    else inf,
                    f"export_{i}_{t}",
                )
                mode[i, t] = solver.BoolVar(f"battery_mode_{i}_{t}")
            for t in range(T + 1):
                soc[i, t] = solver.NumVar(0, usable, f"soc_{i}_{t}")
        for i in key:
            for j in key:
                if i != j:
                    for t in range(T):
                        trades[i, j, t] = solver.NumVar(0, inf, f"trade_{i}_{j}_{t}")
        for i in key:
            load = self.houses[i].consumption_kWh.to_numpy(float) / DT
            pv = self.houses[i].production_kWh.to_numpy(float) / DT
            solver.Add(soc[i, 0] == soc0)
            for t in range(T):
                incoming = solver.Sum(trades[j, i, t] for j in key if j != i)
                outgoing = solver.Sum(trades[i, j, t] for j in key if j != i)
                solver.Add(
                    imports[i, t] + discharge[i, t] + pv[t] + incoming
                    == load[t] + charge[i, t] + exports[i, t] + outgoing
                )
                solver.Add(
                    soc[i, t + 1]
                    == soc[i, t] + eta * charge[i, t] * DT - discharge[i, t] * DT / eta
                )
                solver.Add(charge[i, t] <= cfg.power_kw * mode[i, t])
                solver.Add(discharge[i, t] <= cfg.power_kw * (1 - mode[i, t]))
                if len(key) == 1:
                    grid_mode = solver.BoolVar(f"grid_mode_{i}_{t}")
                    solver.Add(
                        imports[i, t]
                        <= (
                            cfg.grid_import_max_kw
                            if cfg.grid_import_max_kw is not None
                            else 1e6
                        )
                        * grid_mode
                    )
                    solver.Add(
                        exports[i, t]
                        <= (
                            cfg.grid_export_max_kw
                            if cfg.grid_export_max_kw is not None
                            else 1e6
                        )
                        * (1 - grid_mode)
                    )
            if cfg.cyclic_soc:
                solver.Add(soc[i, T] == soc[i, 0])
        energy = solver.Sum(
            (buy[t] * imports[i, t] - sell[t] * exports[i, t]) * DT
            for i in key
            for t in range(T)
        )
        degradation = solver.Sum(
            0.5
            * cfg.blcoe_eur_kwh
            * cfg.degradation_weight
            * (charge[i, t] + discharge[i, t])
            * DT
            for i in key
            for t in range(T)
        )
        trade_penalty = solver.Sum(cfg.trade_epsilon * v for v in trades.values())
        solver.Minimize(
            energy + degradation + trade_penalty + len(key) * cfg.fixed_cost_eur
        )
        started = time.perf_counter()
        status = solver.Solve()
        elapsed = time.perf_counter() - started
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError(
                f"{self.day}, coalition {key}: scheduling did not reach OPTIMAL (status {status})"
            )
        rows = []
        if with_schedule:
            for i in key:
                for t in range(T):
                    rows.append(
                        {
                            "day": self.day,
                            "player_id": i,
                            "period": t + 1,
                            "charge_kw": charge[i, t].solution_value(),
                            "discharge_kw": discharge[i, t].solution_value(),
                            "grid_import_kw": imports[i, t].solution_value(),
                            "grid_export_kw": exports[i, t].solution_value(),
                            "soc_start_kwh": soc[i, t].solution_value(),
                            "soc_end_kwh": soc[i, t + 1].solution_value(),
                            "internal_in_kw": sum(
                                trades[j, i, t].solution_value() for j in key if j != i
                            ),
                            "internal_out_kw": sum(
                                trades[i, j, t].solution_value() for j in key if j != i
                            ),
                        }
                    )
        result = SolveResult(
            float(solver.Objective().Value()),
            float(energy.solution_value()),
            float(degradation.solution_value()),
            float(trade_penalty.solution_value()),
            elapsed,
            pd.DataFrame(rows),
        )
        self.cache[key] = result
        return result


def positive_marginal_allocation(model, members):
    """Allocate coordination surplus in proportion to positive leave-one-out values."""
    members = tuple(sorted(members))
    solo = model.solo_costs()
    surplus = model.value(members)
    if surplus < -1e-6:
        raise RuntimeError(f"Coalition cost exceeds solo cost by {-surplus:.6g} EUR")
    surplus = max(0.0, surplus)
    marginals = np.array(
        [
            model.value(members) - model.value([j for j in members if j != i])
            for i in members
        ]
    )
    positive = np.maximum(marginals, 0)
    weights = (
        positive / positive.sum()
        if positive.sum() > 1e-12
        else np.full(len(members), 1 / len(members))
    )
    return pd.DataFrame(
        [
            {
                "day": model.day,
                "player_id": i,
                "members": ",".join(map(str, members)),
                "solo_cost": solo[i],
                "marginal_contribution": float(marginals[k]),
                "positive_marginal_contribution": float(positive[k]),
                "allocation_weight": float(weights[k]),
                "allocated_bonus": float(surplus * weights[k]),
                "allocated_cost": float(solo[i] - surplus * weights[k]),
            }
            for k, i in enumerate(members)
        ]
    )


def exact_shapley(model, members):
    """Exact cooperation-game Shapley benchmark for a small configured subset."""
    members = tuple(sorted(members))
    n = len(members)
    if not 1 <= n <= 10:
        raise ValueError("exact Shapley supports 1..10 players")
    from math import comb

    values = {}
    for i in members:
        others = [j for j in members if j != i]
        values[i] = sum(
            (model.value((*subset, i)) - model.value(subset)) / (n * comb(n - 1, k))
            for k in range(n)
            for subset in combinations(others, k)
        )
    return values
