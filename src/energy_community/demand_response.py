"""Targeted demand response with the thesis ideal central-battery benchmark.

Adapted from 09_demand_response_targeted_load_shift.py. Uses perfect efficiency,
no power/degradation limits, and empty initial/final SOC as in that experiment.
"""

import time
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

T = 96
DT = 0.25
INITIAL_SOC_KWH = 0.0
FINAL_SOC_KWH = 0.0
THROUGHPUT_TIE_BREAKER_EUR_PER_KWH = 1e-9
EPS = 1e-10


def safe_pct(numerator, denominator):
    if abs(float(denominator)) <= EPS:
        return np.nan
    return 100.0 * float(numerator) / float(denominator)


def period_to_time_label(period):
    minutes = (int(period) - 1) * 15
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def rolling_window_candidates(window_periods):
    window_periods = int(window_periods)
    for start in range(1, T - window_periods + 2):
        end = start + window_periods - 1
        yield start, end


def windows_overlap(start_a, end_a, start_b, end_b):
    return max(int(start_a), int(start_b)) <= min(int(end_a), int(end_b))


def compute_no_battery_cost(aggregate):
    net = aggregate["net_load_kWh"].to_numpy(dtype=float)
    buy = aggregate["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = aggregate["sell_price_eur_kwh"].to_numpy(dtype=float)

    import_kwh = np.maximum(net, 0.0)
    export_kwh = np.maximum(-net, 0.0)
    cost = float(np.sum(buy * import_kwh - sell * export_kwh))

    return cost, import_kwh, export_kwh


def solve_ideal_battery_day(aggregate, capacity_kwh):
    capacity_kwh = float(capacity_kwh)
    no_batt_cost, no_batt_import, no_batt_export = compute_no_battery_cost(aggregate)

    if capacity_kwh <= EPS:
        schedule = pd.DataFrame(
            {
                "period": aggregate["period"].to_numpy(dtype=int),
                "battery_capacity_kwh": 0.0,
                "consumption_kWh": aggregate["consumption_kWh"].to_numpy(dtype=float),
                "production_kWh": aggregate["production_kWh"].to_numpy(dtype=float),
                "net_load_kWh": aggregate["net_load_kWh"].to_numpy(dtype=float),
                "buy_price_eur_kwh": aggregate["buy_price_eur_kwh"].to_numpy(
                    dtype=float
                ),
                "sell_price_eur_kwh": aggregate["sell_price_eur_kwh"].to_numpy(
                    dtype=float
                ),
                "battery_charge_kWh": np.zeros(T, dtype=float),
                "battery_discharge_kWh": np.zeros(T, dtype=float),
                "battery_soc_start_kWh": np.zeros(T, dtype=float),
                "battery_soc_end_kWh": np.zeros(T, dtype=float),
                "grid_import_kWh": no_batt_import,
                "grid_export_kWh": no_batt_export,
            }
        )
        return {
            "cost_eur": no_batt_cost,
            "grid_import_kwh": float(no_batt_import.sum()),
            "grid_export_kwh": float(no_batt_export.sum()),
            "battery_charged_kwh": 0.0,
            "battery_discharged_kwh": 0.0,
            "battery_throughput_kwh": 0.0,
            "battery_max_soc_kwh": 0.0,
            "solve_time_sec": 0.0,
            "solver_name": "closed_form_no_battery",
            "schedule": schedule,
        }

    solver = pywraplp.Solver.CreateSolver("GLOP")
    solver_name = "GLOP"
    if not solver:
        solver = pywraplp.Solver.CreateSolver("CLP")
        solver_name = "CLP"
    if not solver:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        solver_name = "SCIP"
    if not solver:
        raise RuntimeError("No OR-Tools linear solver is available.")

    inf = solver.infinity()
    consumption = aggregate["consumption_kWh"].to_numpy(dtype=float)
    production = aggregate["production_kWh"].to_numpy(dtype=float)
    buy = aggregate["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = aggregate["sell_price_eur_kwh"].to_numpy(dtype=float)

    charge = [solver.NumVar(0.0, capacity_kwh, f"charge_{t}") for t in range(T)]
    discharge = [solver.NumVar(0.0, capacity_kwh, f"discharge_{t}") for t in range(T)]
    grid_import = [solver.NumVar(0.0, inf, f"grid_import_{t}") for t in range(T)]
    grid_export = [solver.NumVar(0.0, inf, f"grid_export_{t}") for t in range(T)]
    soc = [solver.NumVar(0.0, capacity_kwh, f"soc_{t}") for t in range(T + 1)]

    solver.Add(soc[0] == INITIAL_SOC_KWH)

    for t in range(T):
        solver.Add(
            grid_import[t] + discharge[t] + production[t]
            == consumption[t] + charge[t] + grid_export[t]
        )
        solver.Add(soc[t + 1] == soc[t] + charge[t] - discharge[t])

    solver.Add(soc[T] == FINAL_SOC_KWH)

    energy_cost = solver.Sum(
        buy[t] * grid_import[t] - sell[t] * grid_export[t] for t in range(T)
    )
    tie_breaker = solver.Sum(
        THROUGHPUT_TIE_BREAKER_EUR_PER_KWH * (charge[t] + discharge[t])
        for t in range(T)
    )
    solver.Minimize(energy_cost + tie_breaker)

    start_time = time.perf_counter()
    status = solver.Solve()
    solve_time = time.perf_counter() - start_time

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Ideal battery optimization not optimal. Status={status}")

    charge_values = np.array(
        [charge[t].solution_value() for t in range(T)], dtype=float
    )
    discharge_values = np.array(
        [discharge[t].solution_value() for t in range(T)], dtype=float
    )
    import_values = np.array(
        [grid_import[t].solution_value() for t in range(T)], dtype=float
    )
    export_values = np.array(
        [grid_export[t].solution_value() for t in range(T)], dtype=float
    )
    soc_values = np.array([soc[t].solution_value() for t in range(T + 1)], dtype=float)

    market_cost = float(np.sum(buy * import_values - sell * export_values))
    throughput_kwh = float(0.5 * (charge_values.sum() + discharge_values.sum()))

    schedule = pd.DataFrame(
        {
            "period": aggregate["period"].to_numpy(dtype=int),
            "battery_capacity_kwh": capacity_kwh,
            "consumption_kWh": consumption,
            "production_kWh": production,
            "net_load_kWh": aggregate["net_load_kWh"].to_numpy(dtype=float),
            "buy_price_eur_kwh": buy,
            "sell_price_eur_kwh": sell,
            "battery_charge_kWh": charge_values,
            "battery_discharge_kWh": discharge_values,
            "battery_soc_start_kWh": soc_values[:-1],
            "battery_soc_end_kWh": soc_values[1:],
            "grid_import_kWh": import_values,
            "grid_export_kWh": export_values,
        }
    )

    return {
        "cost_eur": market_cost,
        "grid_import_kwh": float(import_values.sum()),
        "grid_export_kwh": float(export_values.sum()),
        "battery_charged_kwh": float(charge_values.sum()),
        "battery_discharged_kwh": float(discharge_values.sum()),
        "battery_throughput_kwh": throughput_kwh,
        "battery_max_soc_kwh": float(soc_values.max()),
        "solve_time_sec": float(solve_time),
        "solver_name": solver_name,
        "schedule": schedule,
    }


def window_values_for_activation_metric(aggregate, metric):
    net = aggregate["net_load_kWh"].to_numpy(dtype=float)
    consumption = aggregate["consumption_kWh"].to_numpy(dtype=float)
    buy = aggregate["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = aggregate["sell_price_eur_kwh"].to_numpy(dtype=float)

    if metric == "max_import_cost":
        return buy * np.maximum(net, 0.0)
    if metric == "max_net_cost":
        return buy * np.maximum(net, 0.0) - sell * np.maximum(-net, 0.0)
    if metric == "max_net_load":
        return net
    if metric == "max_consumption":
        return consumption

    raise ValueError(
        "Unknown activation metric. Use one of: "
        "max_import_cost, max_net_cost, max_net_load, max_consumption."
    )


def select_activation_window(aggregate, window_periods, metric):
    values = window_values_for_activation_metric(aggregate, metric)
    net = aggregate["net_load_kWh"].to_numpy(dtype=float)
    buy = aggregate["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = aggregate["sell_price_eur_kwh"].to_numpy(dtype=float)
    consumption = aggregate["consumption_kWh"].to_numpy(dtype=float)

    best = None
    best_score = -np.inf
    for start, end in rolling_window_candidates(window_periods):
        idx = slice(start - 1, end)
        score = float(values[idx].sum())
        if score > best_score:
            best_score = score
            best = (start, end)

    if best is None:
        return None

    start, end = best
    idx = slice(start - 1, end)
    import_kwh = float(np.maximum(net[idx], 0.0).sum())
    export_kwh = float(np.maximum(-net[idx], 0.0).sum())
    import_cost = float(np.sum(buy[idx] * np.maximum(net[idx], 0.0)))
    net_cost = float(
        np.sum(
            buy[idx] * np.maximum(net[idx], 0.0)
            - sell[idx] * np.maximum(-net[idx], 0.0)
        )
    )

    return {
        "activation_start_period": int(start),
        "activation_end_period": int(end),
        "activation_start_time": period_to_time_label(start),
        "activation_end_time": period_to_time_label(end + 1),
        "activation_metric": metric,
        "activation_metric_value": float(best_score),
        "activation_import_kwh": import_kwh,
        "activation_export_kwh": export_kwh,
        "activation_consumption_kwh": float(consumption[idx].sum()),
        "activation_import_cost_eur": import_cost,
        "activation_net_cost_eur": net_cost,
        "activation_mean_buy_price_eur_kwh": float(buy[idx].mean()),
        "activation_mean_sell_price_eur_kwh": float(sell[idx].mean()),
    }


def recovery_candidates(activation_start, activation_end, window_periods, allow_before):
    for start, end in rolling_window_candidates(window_periods):
        if windows_overlap(start, end, activation_start, activation_end):
            continue
        if not allow_before and end < activation_start:
            continue
        yield start, end


def baseline_marginal_consumption_cost(schedule):
    buy = schedule["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = schedule["sell_price_eur_kwh"].to_numpy(dtype=float)
    export = schedule["grid_export_kWh"].to_numpy(dtype=float)
    return np.where(export > EPS, sell, buy)


def select_recovery_window(
    aggregate,
    baseline_schedule,
    capacity_kwh,
    activation,
    window_periods,
    mode,
    allow_before,
):
    activation_start = int(activation["activation_start_period"])
    activation_end = int(activation["activation_end_period"])

    buy = aggregate["buy_price_eur_kwh"].to_numpy(dtype=float)
    sell = aggregate["sell_price_eur_kwh"].to_numpy(dtype=float)
    net = aggregate["net_load_kWh"].to_numpy(dtype=float)
    schedule = baseline_schedule.copy()
    marginal = baseline_marginal_consumption_cost(schedule)
    grid_export = (
        schedule["grid_export_KWh"].to_numpy(dtype=float)
        if "grid_export_KWh" in schedule.columns
        else schedule["grid_export_kWh"].to_numpy(dtype=float)
    )
    soc_end = (
        schedule["battery_soc_end_KWh"].to_numpy(dtype=float)
        if "battery_soc_end_KWh" in schedule.columns
        else schedule["battery_soc_end_kWh"].to_numpy(dtype=float)
    )
    if float(capacity_kwh) > EPS:
        full_soc = soc_end >= float(capacity_kwh) - 1e-8
    else:
        full_soc = np.zeros_like(soc_end, dtype=bool)

    best_tuple = None
    best_window = None
    for start, end in recovery_candidates(
        activation_start,
        activation_end,
        window_periods,
        allow_before,
    ):
        idx = slice(start - 1, end)
        mean_buy = float(buy[idx].mean())
        mean_sell = float(sell[idx].mean())
        mean_marginal = float(marginal[idx].mean())
        export_kwh = float(grid_export[idx].sum())
        pre_battery_surplus_kwh = float(np.maximum(-net[idx], 0.0).sum())
        full_soc_share = float(full_soc[idx].mean()) if len(full_soc[idx]) else 0.0

        if mode == "cheapest_price":
            rank_tuple = (mean_buy, mean_marginal, -export_kwh, start)
        elif mode == "battery_aware":
            rank_tuple = (mean_marginal, mean_buy, -export_kwh, -full_soc_share, start)
        elif mode == "surplus_first":
            rank_tuple = (-export_kwh, mean_marginal, mean_buy, -full_soc_share, start)
        else:
            raise ValueError(
                "Unknown recovery mode. Use one of: cheapest_price, battery_aware, surplus_first."
            )

        if best_tuple is None or rank_tuple < best_tuple:
            best_tuple = rank_tuple
            best_window = {
                "recovery_mode": mode,
                "recovery_start_period": int(start),
                "recovery_end_period": int(end),
                "recovery_start_time": period_to_time_label(start),
                "recovery_end_time": period_to_time_label(end + 1),
                "recovery_before_activation": bool(end < activation_start),
                "recovery_mean_buy_price_eur_kwh": mean_buy,
                "recovery_mean_sell_price_eur_kwh": mean_sell,
                "recovery_mean_post_battery_marginal_cost_eur_kwh": mean_marginal,
                "recovery_baseline_grid_export_kwh": export_kwh,
                "recovery_pre_battery_surplus_kwh": pre_battery_surplus_kwh,
                "recovery_full_soc_share": full_soc_share,
            }

    if best_window is None:
        raise ValueError("No feasible recovery window was found.")
    return best_window


def apply_fractional_load_shift(
    aggregate,
    player_frames,
    activation,
    recovery,
    reduction_fraction,
):
    activation_periods = list(
        range(
            int(activation["activation_start_period"]),
            int(activation["activation_end_period"]) + 1,
        )
    )
    recovery_periods = list(
        range(
            int(recovery["recovery_start_period"]),
            int(recovery["recovery_end_period"]) + 1,
        )
    )
    activation_idx = np.array([p - 1 for p in activation_periods], dtype=int)
    recovery_idx = np.array([p - 1 for p in recovery_periods], dtype=int)

    removed_by_period = np.zeros(T, dtype=float)
    added_by_period = np.zeros(T, dtype=float)
    player_rows = []

    for player_id, frame in player_frames.items():
        consumption = frame["consumption_kWh"].to_numpy(dtype=float)
        activation_consumption = consumption[activation_idx]
        reduction = float(reduction_fraction) * activation_consumption
        shifted_kwh = float(reduction.sum())
        if shifted_kwh <= EPS:
            continue

        addition = np.full(
            len(recovery_idx), shifted_kwh / len(recovery_idx), dtype=float
        )
        removed_by_period[activation_idx] += reduction
        added_by_period[recovery_idx] += addition

        player_rows.append(
            {
                "player_id": int(player_id),
                "activation_consumption_kwh": float(activation_consumption.sum()),
                "shifted_kwh": shifted_kwh,
                "dr_strategy": "fractional",
                "dr_parameter": float(reduction_fraction),
                "reduction_fraction": float(reduction_fraction),
                "cap_ratio": np.nan,
                "activation_player_cap_kwh": np.nan,
                "activation_mean_player_consumption_kwh": np.nan,
                "capped_by_dr": bool(shifted_kwh > EPS),
                "recovery_added_kwh": float(addition.sum()),
            }
        )

    modified = aggregate.copy()
    modified_consumption = (
        aggregate["consumption_kWh"].to_numpy(dtype=float)
        - removed_by_period
        + added_by_period
    )

    if np.any(modified_consumption < -1e-8):
        raise ValueError("DR modification produced negative aggregate consumption.")

    modified["consumption_kWh"] = np.maximum(modified_consumption, 0.0)
    modified["net_load_kWh"] = modified["consumption_kWh"] - modified["production_kWh"]

    shift_profile = pd.DataFrame(
        {
            "period": aggregate["period"].to_numpy(dtype=int),
            "removed_kWh": removed_by_period,
            "added_kWh": added_by_period,
            "net_shift_kWh": added_by_period - removed_by_period,
            "baseline_consumption_kWh": aggregate["consumption_kWh"].to_numpy(
                dtype=float
            ),
            "dr_consumption_kWh": modified["consumption_kWh"].to_numpy(dtype=float),
            "baseline_net_load_kWh": aggregate["net_load_kWh"].to_numpy(dtype=float),
            "dr_net_load_kWh": modified["net_load_kWh"].to_numpy(dtype=float),
        }
    )

    player_shift = pd.DataFrame(player_rows)
    return modified, shift_profile, player_shift


def apply_cap_load_shift(
    aggregate,
    player_frames,
    activation,
    recovery,
    cap_ratio,
):
    activation_periods = list(
        range(
            int(activation["activation_start_period"]),
            int(activation["activation_end_period"]) + 1,
        )
    )
    recovery_periods = list(
        range(
            int(recovery["recovery_start_period"]),
            int(recovery["recovery_end_period"]) + 1,
        )
    )
    activation_idx = np.array([p - 1 for p in activation_periods], dtype=int)
    recovery_idx = np.array([p - 1 for p in recovery_periods], dtype=int)

    activation_totals = []
    for frame in player_frames.values():
        consumption = frame["consumption_kWh"].to_numpy(dtype=float)
        activation_totals.append(float(consumption[activation_idx].sum()))

    mean_activation_consumption = (
        float(np.mean(activation_totals)) if activation_totals else 0.0
    )
    player_cap_kwh = float(cap_ratio) * mean_activation_consumption

    removed_by_period = np.zeros(T, dtype=float)
    added_by_period = np.zeros(T, dtype=float)
    player_rows = []

    for player_id, frame in player_frames.items():
        consumption = frame["consumption_kWh"].to_numpy(dtype=float)
        activation_consumption = consumption[activation_idx]
        activation_total = float(activation_consumption.sum())

        shifted_kwh = max(0.0, activation_total - player_cap_kwh)
        if shifted_kwh <= EPS or activation_total <= EPS:
            player_rows.append(
                {
                    "player_id": int(player_id),
                    "activation_consumption_kwh": activation_total,
                    "shifted_kwh": 0.0,
                    "dr_strategy": "cap",
                    "dr_parameter": float(cap_ratio),
                    "reduction_fraction": np.nan,
                    "cap_ratio": float(cap_ratio),
                    "activation_player_cap_kwh": player_cap_kwh,
                    "activation_mean_player_consumption_kwh": mean_activation_consumption,
                    "capped_by_dr": False,
                    "recovery_added_kwh": 0.0,
                }
            )
            continue

        reduction_fraction_for_player = shifted_kwh / activation_total
        reduction = reduction_fraction_for_player * activation_consumption
        addition = np.full(
            len(recovery_idx), shifted_kwh / len(recovery_idx), dtype=float
        )
        removed_by_period[activation_idx] += reduction
        added_by_period[recovery_idx] += addition

        player_rows.append(
            {
                "player_id": int(player_id),
                "activation_consumption_kwh": activation_total,
                "shifted_kwh": shifted_kwh,
                "dr_strategy": "cap",
                "dr_parameter": float(cap_ratio),
                "reduction_fraction": reduction_fraction_for_player,
                "cap_ratio": float(cap_ratio),
                "activation_player_cap_kwh": player_cap_kwh,
                "activation_mean_player_consumption_kwh": mean_activation_consumption,
                "capped_by_dr": True,
                "recovery_added_kwh": float(addition.sum()),
            }
        )

    modified = aggregate.copy()
    modified_consumption = (
        aggregate["consumption_kWh"].to_numpy(dtype=float)
        - removed_by_period
        + added_by_period
    )

    if np.any(modified_consumption < -1e-8):
        raise ValueError("DR modification produced negative aggregate consumption.")

    modified["consumption_kWh"] = np.maximum(modified_consumption, 0.0)
    modified["net_load_kWh"] = modified["consumption_kWh"] - modified["production_kWh"]

    shift_profile = pd.DataFrame(
        {
            "period": aggregate["period"].to_numpy(dtype=int),
            "removed_kWh": removed_by_period,
            "added_kWh": added_by_period,
            "net_shift_kWh": added_by_period - removed_by_period,
            "baseline_consumption_kWh": aggregate["consumption_kWh"].to_numpy(
                dtype=float
            ),
            "dr_consumption_kWh": modified["consumption_kWh"].to_numpy(dtype=float),
            "baseline_net_load_kWh": aggregate["net_load_kWh"].to_numpy(dtype=float),
            "dr_net_load_kWh": modified["net_load_kWh"].to_numpy(dtype=float),
        }
    )

    player_shift = pd.DataFrame(player_rows)
    return modified, shift_profile, player_shift
