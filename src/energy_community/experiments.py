"""Experiment stages sharing one dataset, model, output directory and cost cache."""

from dataclasses import asdict
from pathlib import Path
from itertools import combinations
import hashlib
import json
import os
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from .config import Config
from .data import Dataset, generate_demo
from .model import SolveResult, positive_marginal_allocation, exact_shapley
from .clustering import (
    standardize_matrix,
    pairwise_euclidean_distance_matrix,
    run_kmedoids,
    remap_clusters_by_medoid_day,
)
from .formation import (
    exact_grouping,
    local_grouping,
    balanced_random_partition,
    formation_table,
)
from . import demand_response as dr


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class Experiment:
    def __init__(self, config: Config, base_dir, output=None):
        self.config = config.validate()
        self.base_dir = Path(base_dir).resolve()
        self.output = (
            Path(output).resolve()
            if output
            else (Path(base_dir) / config.output_dir).resolve()
        )
        self.output.mkdir(parents=True, exist_ok=True)
        if config.demo:
            source = self.output / "demo_data"
            generate_demo(source, config.players, config.demo_days, config.seed)
        else:
            source = (Path(base_dir) / config.data_dir).resolve()
        self.dataset = Dataset(
            source,
            config.players,
            config.repair_small_gaps,
            config.start_day,
            config.end_day,
        )
        from . import __version__

        signature = {
            "config": asdict(config),
            "input_sha256": self.dataset.hashes,
            "framework_version": __version__,
            "model_cache_version": 1,
        }
        if config.partition_file:
            from .data import digest_file

            signature["partition_sha256"] = digest_file(
                self.base_dir / config.partition_file
            )
        self.fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        self.manifest = self.output / "manifest.json"
        if self.manifest.exists():
            old = json.loads(self.manifest.read_text())
            if old["fingerprint"] != self.fingerprint:
                raise ValueError(
                    "This output directory belongs to different settings or inputs. Select another directory with --output."
                )
        self.models = {}
        self.groups = None
        self.completed = []
        save_json(
            self.manifest,
            {
                **signature,
                "fingerprint": self.fingerprint,
                "days": self.dataset.days,
                "data_kind": "synthetic_demo"
                if config.demo
                else "user_supplied_prepared_csv",
                "python": platform.python_version(),
                "dependencies": {
                    p: version(p) for p in ["numpy", "pandas", "matplotlib", "ortools"]
                },
                "completed_stages": json.loads(self.manifest.read_text()).get(
                    "completed_stages", []
                )
                if self.manifest.exists()
                else [],
            },
        )

    def model(self, day):
        if day not in self.models:
            model = self.dataset.day_model(day, self.config.battery)
            path = self.output / "cache" / f"{day}.json"
            if path.exists():
                for key, values in json.loads(path.read_text()).items():
                    model.cache[tuple(map(int, key.split(",")))] = SolveResult(
                        **values, schedule=pd.DataFrame()
                    )
            self.models[day] = model
        return self.models[day]

    def checkpoint(self, day):
        model = self.models[day]
        values = {
            ",".join(map(str, key)): {
                k: getattr(result, k)
                for k in [
                    "cost",
                    "energy_cost",
                    "degradation_cost",
                    "trade_penalty",
                    "solve_seconds",
                ]
            }
            for key, result in model.cache.items()
        }
        save_json(self.output / "cache" / f"{day}.json", values)

    def record_stage(self, name):
        data = json.loads(self.manifest.read_text())
        data["completed_stages"] = list(
            dict.fromkeys([*data["completed_stages"], name])
        )
        save_json(self.manifest, data)

    def invalidate_stage(self, name):
        """An interrupted rerun cannot be mistaken for an earlier completed stage."""
        invalid = {name}
        if name != "report":
            invalid.add("report")
        while True:
            downstream = {
                stage
                for stage, deps in DEPENDENCIES.items()
                if any(dep in invalid for dep in deps)
            }
            updated = invalid | downstream
            if updated == invalid:
                break
            invalid = updated
        data = json.loads(self.manifest.read_text())
        data["completed_stages"] = [
            stage for stage in data["completed_stages"] if stage not in invalid
        ]
        save_json(self.manifest, data)

    def prepare(self):
        self.dataset.save_prepared(self.output / "prepared_data")
        pd.DataFrame(self.dataset.audit).to_csv(
            self.output / "data_audit.csv", index=False
        )
        pd.DataFrame({"day": self.dataset.days}).to_csv(
            self.output / "valid_days.csv", index=False
        )
        print(
            f"Prepared {len(self.dataset.days)} common days for {len(self.config.players)} players",
            flush=True,
        )

    def representative(self):
        days = self.dataset.days
        if len(days) < 3:
            raise ValueError(
                "Representative-day selection needs at least three common days"
            )
        profiles = np.array(
            [
                sum(
                    (
                        self.dataset.frames[i][d].net_load_kWh.to_numpy(float)
                        for i in self.config.players
                    ),
                    start=np.zeros(96),
                )
                for d in days
            ]
        )
        features = pd.DataFrame(
            {
                "day": days,
                "positive_net_load_kWh": np.maximum(profiles, 0).sum(axis=1),
                "surplus_net_load_kWh": np.maximum(-profiles, 0).sum(axis=1),
            }
        )
        imp = int(np.argmax(features.positive_net_load_kWh))
        surplus = next(
            int(k)
            for k in np.argsort(
                -features.surplus_net_load_kWh.to_numpy(), kind="stable"
            )
            if k != imp
        )
        extremes = {imp: "import_extreme", surplus: "surplus_extreme"}
        remaining = [k for k in range(len(days)) if k not in extremes]
        x = profiles[remaining]
        if self.config.representative.standardize:
            x, _, _ = standardize_matrix(x)
        r = self.config.representative
        k = min(r.clusters, len(remaining))
        best, log = run_kmedoids(
            pairwise_euclidean_distance_matrix(x),
            k,
            r.initializations,
            r.max_iterations,
            self.config.seed,
        )
        remaining_days = [days[idx] for idx in remaining]
        labels, medoids = remap_clusters_by_medoid_day(
            best["labels"], best["medoids"], remaining_days
        )
        rows = [
            {"day": days[idx], "kind": kind, "weight": 1}
            for idx, kind in extremes.items()
        ]
        rows += [
            {
                "day": row.representative_day,
                "kind": "medoid",
                "weight": int((labels == row.cluster_id).sum()),
            }
            for row in medoids.itertuples()
        ]
        reps = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
        if (
            reps.day.duplicated().any()
            or int(reps.weight.sum()) != len(days)
            or (reps.weight <= 0).any()
        ):
            raise RuntimeError(
                "Representative days must be distinct and cover the full common-day weight"
            )
        reps.to_csv(self.output / "representative_days.csv", index=False)
        features.to_csv(self.output / "daily_net_load_features.csv", index=False)
        log.to_csv(self.output / "kmedoids_initializations.csv", index=False)
        pd.DataFrame({"day": remaining_days, "cluster": labels}).to_csv(
            self.output / "cluster_assignments.csv", index=False
        )
        np.save(self.output / "aggregate_net_load_profiles.npy", profiles)
        print(
            f"Selected {len(reps)} representative days, total weight {reps.weight.sum()}",
            flush=True,
        )

    def scores(self):
        reps = pd.read_csv(self.output / "representative_days.csv")
        rows = []
        for day_idx, row in enumerate(reps.itertuples(), start=1):
            model = self.model(row.day)
            solo = model.solo_costs()
            for i, j in combinations(self.config.players, 2):
                result = model.solve([i, j])
                gain = solo[i] + solo[j] - result.cost
                rows.append(
                    {
                        "day": row.day,
                        "weight": row.weight,
                        "player_i": i,
                        "player_j": j,
                        "pair_cost": result.cost,
                        "pairwise_gain": gain,
                        "weighted_pairwise_gain": row.weight * gain,
                    }
                )
                self.checkpoint(row.day)
            pd.DataFrame(rows).to_csv(self.output / "pairwise_daily.csv", index=False)
            print(
                f"Pairwise scores: representative day {day_idx}/{len(reps)}", flush=True
            )
        long = (
            pd.DataFrame(rows)
            .groupby(["player_i", "player_j"], as_index=False)
            .weighted_pairwise_gain.sum()
        )
        matrix = pd.DataFrame(
            0.0, index=self.config.players, columns=self.config.players
        )
        for row in long.itertuples():
            matrix.loc[row.player_i, row.player_j] = matrix.loc[
                row.player_j, row.player_i
            ] = row.weighted_pairwise_gain
        matrix.index.name = "player_id"
        matrix.to_csv(self.output / "pairwise_scores.csv")
        long.to_csv(self.output / "pairwise_scores_long.csv", index=False)

    def score_matrix(self):
        matrix = pd.read_csv(self.output / "pairwise_scores.csv", index_col=0)
        matrix.index = matrix.index.astype(int)
        matrix.columns = matrix.columns.astype(int)
        return matrix

    def formation(self):
        f = self.config.formation
        if self.config.partition_file:
            table = pd.read_csv(self.base_dir / self.config.partition_file)
            groups = [list(map(int, str(m).split(","))) for m in table.members]
            if sorted(p for g in groups for p in g) != sorted(
                self.config.players
            ) or any(len(g) != f.group_size for g in groups):
                raise ValueError(
                    "partition_file must cover every configured player exactly once in equal-size groups"
                )
            self.groups = groups
            table.to_csv(self.output / "selected_subcoalitions.csv", index=False)
            save_json(
                self.output / "formation_summary.json",
                {"method": "supplied_partition", "source": self.config.partition_file},
            )
            print(f"Using supplied partition: {self.groups}", flush=True)
            return
        scores = self.score_matrix()
        method = (
            f.method
            if f.method != "auto"
            else (
                "milp"
                if len(self.config.players) <= f.exact_max_players
                else "local_search"
            )
        )
        if method == "milp":
            self.groups, meta = exact_grouping(
                self.config.players, scores, f.group_size, f.time_limit_seconds
            )
        else:
            self.groups, meta, log = local_grouping(
                self.config.players,
                scores,
                f.group_size,
                f.local_starts,
                f.local_passes,
                self.config.seed,
                f.local_patience,
            )
            log.to_csv(self.output / "local_search_starts.csv", index=False)
        formation_table(self.groups, scores).to_csv(
            self.output / "selected_subcoalitions.csv", index=False
        )
        save_json(self.output / "formation_summary.json", meta)
        print(
            f"Formed {len(self.groups)} groups using {method}: {self.groups}",
            flush=True,
        )

    def load_groups(self):
        if self.groups is None:
            table = pd.read_csv(self.output / "selected_subcoalitions.csv")
            self.groups = [list(map(int, str(s).split(","))) for s in table.members]
        return self.groups

    def evaluate(self):
        groups = self.load_groups()
        f = self.config.formation
        random_groups = [
            balanced_random_partition(
                self.config.players,
                self.dataset.prosumers(),
                f.group_size,
                self.config.seed + k,
            )
            for k in range(f.random_partitions)
        ]
        save_json(self.output / "random_partitions.json", random_groups)
        rows, solo_rows, group_rows, random_rows = [], [], [], []
        for day_idx, day in enumerate(self.dataset.days, start=1):
            model = self.model(day)
            solo = model.solo_costs()
            baseline = model.baseline()
            solo_total = sum(solo.values())
            grand = model.solve(self.config.players).cost
            subtotal = 0.0
            for k, members in enumerate(groups, start=1):
                cost = model.solve(members).cost
                subtotal += cost
                group_rows.append(
                    {
                        "day": day,
                        "coalition_id": k,
                        "members": ",".join(map(str, members)),
                        "cost_eur": cost,
                        "coordination_gain_eur": sum(solo[i] for i in members) - cost,
                    }
                )
            rows.append(
                {
                    "day": day,
                    "baseline_cost_eur": baseline,
                    "solo_cost_eur": solo_total,
                    "grand_cost_eur": grand,
                    "subcoalitions_cost_eur": subtotal,
                    "grand_coordination_gain_eur": solo_total - grand,
                    "subcoalitions_coordination_gain_eur": solo_total - subtotal,
                }
            )
            solo_rows += [
                {"day": day, "player_id": i, "solo_cost_eur": solo[i]}
                for i in self.config.players
            ]
            for k, partition in enumerate(random_groups, start=1):
                cost = sum(model.solve(members).cost for members in partition)
                random_rows.append(
                    {
                        "day": day,
                        "partition_id": k,
                        "cost_eur": cost,
                        "coordination_gain_eur": solo_total - cost,
                    }
                )
            self.checkpoint(day)
            pd.DataFrame(rows).to_csv(self.output / "daily_evaluation.csv", index=False)
            print(f"Evaluation: {day_idx}/{len(self.dataset.days)} ({day})", flush=True)
        pd.DataFrame(solo_rows).to_csv(
            self.output / "daily_solo_costs.csv", index=False
        )
        pd.DataFrame(group_rows).to_csv(
            self.output / "daily_subcoalition_costs.csv", index=False
        )
        pd.DataFrame(
            random_rows,
            columns=["day", "partition_id", "cost_eur", "coordination_gain_eur"],
        ).to_csv(self.output / "random_evaluation.csv", index=False)
        totals = pd.DataFrame(rows).drop(columns="day").sum().to_dict()
        gain = totals["grand_coordination_gain_eur"]
        totals["coordination_gain_recovery_pct"] = (
            100 * totals["subcoalitions_coordination_gain_eur"] / gain
            if abs(gain) > 1e-10
            else None
        )
        save_json(
            self.output / "evaluation_summary.json",
            {"number_of_days": len(rows), **totals},
        )

    def allocation(self):
        rows = []
        for day_idx, day in enumerate(self.dataset.days, start=1):
            model = self.model(day)
            for k, members in enumerate(self.load_groups(), start=1):
                allocation = positive_marginal_allocation(model, members)
                allocation["coalition_id"] = k
                rows.extend(allocation.to_dict("records"))
            self.checkpoint(day)
            pd.DataFrame(rows).to_csv(
                self.output / "daily_allocations.csv", index=False
            )
            print(f"Profit allocation: {day_idx}/{len(self.dataset.days)}", flush=True)
        table = (
            pd.DataFrame(rows)
            .groupby(["coalition_id", "player_id"], as_index=False)[
                ["solo_cost", "allocated_bonus", "allocated_cost"]
            ]
            .sum()
        )
        table.to_csv(self.output / "annual_allocations.csv", index=False)

    def shapley(self):
        players = self.config.shapley_players
        if not players:
            print("Shapley benchmark disabled (shapley_players is empty)", flush=True)
            return
        days = [
            d
            for d in self.dataset.days
            if (not self.config.shapley_start_day or d >= self.config.shapley_start_day)
            and (not self.config.shapley_end_day or d <= self.config.shapley_end_day)
        ]
        if not days:
            raise ValueError("No common days in the configured Shapley horizon")
        rows = []
        for day_idx, day in enumerate(days, start=1):
            model = self.model(day)
            proposed = positive_marginal_allocation(model, players).set_index(
                "player_id"
            )
            benchmark = exact_shapley(model, players)
            for i in players:
                rows.append(
                    {
                        "day": day,
                        "player_id": i,
                        "proposed_bonus_eur": proposed.loc[i, "allocated_bonus"],
                        "shapley_bonus_eur": benchmark[i],
                        "cooperation_surplus_eur": model.value(players),
                    }
                )
            self.checkpoint(day)
            pd.DataFrame(rows).to_csv(
                self.output / "shapley_comparison.csv", index=False
            )
            print(
                f"Exact Shapley: {day_idx}/{len(days)}, {len(players)} players",
                flush=True,
            )

    def stability(self):
        current = pd.read_csv(self.output / "annual_allocations.csv").set_index(
            "player_id"
        )
        groups = self.load_groups()
        weakest = [
            min(group, key=lambda i: (current.loc[i, "allocated_bonus"], i))
            for group in groups
        ]
        pd.DataFrame(
            {"coalition_id": range(1, len(groups) + 1), "weakest_player": weakest}
        ).to_csv(self.output / "weakest_players.csv", index=False)
        rows, impacts = [], []
        for source, mover in enumerate(weakest):
            for target, members in enumerate(groups):
                if source == target:
                    continue
                new_members = members + [mover]
                bonus = {i: 0.0 for i in new_members}
                for day in self.dataset.days:
                    model = self.model(day)
                    allocation = positive_marginal_allocation(model, new_members)
                    for row in allocation.itertuples():
                        bonus[row.player_id] += row.allocated_bonus
                    self.checkpoint(day)
                delta = bonus[mover] - float(current.loc[mover, "allocated_bonus"])
                target_deltas = {
                    i: bonus[i] - float(current.loc[i, "allocated_bonus"])
                    for i in members
                }
                accepts = all(v >= -1e-8 for v in target_deltas.values())
                rows.append(
                    {
                        "source_coalition_id": source + 1,
                        "target_coalition_id": target + 1,
                        "moving_player": mover,
                        "moving_player_bonus_delta_eur": delta,
                        "player_wants_move": delta > 1e-8,
                        "target_accepts": accepts,
                        "accepted_profitable_move": bool(delta > 1e-8 and accepts),
                    }
                )
                impacts += [
                    {
                        "source_coalition_id": source + 1,
                        "target_coalition_id": target + 1,
                        "moving_player": mover,
                        "target_member": i,
                        "bonus_delta_eur": v,
                    }
                    for i, v in target_deltas.items()
                ]
                pd.DataFrame(rows).to_csv(
                    self.output / "migration_candidates.csv", index=False
                )
                print(
                    f"Migration candidate: {mover} from {source + 1} to {target + 1}",
                    flush=True,
                )
        pd.DataFrame(
            rows,
            columns=[
                "source_coalition_id",
                "target_coalition_id",
                "moving_player",
                "moving_player_bonus_delta_eur",
                "player_wants_move",
                "target_accepts",
                "accepted_profitable_move",
            ],
        ).to_csv(self.output / "migration_candidates.csv", index=False)
        pd.DataFrame(
            impacts,
            columns=[
                "source_coalition_id",
                "target_coalition_id",
                "moving_player",
                "target_member",
                "bonus_delta_eur",
            ],
        ).to_csv(self.output / "migration_target_impacts.csv", index=False)
        save_json(
            self.output / "stability_summary.json",
            {
                "criterion": "weakest-player accepted single-player migration",
                "candidate_moves": len(rows),
                "accepted_profitable_moves": sum(
                    r["accepted_profitable_move"] for r in rows
                ),
                "days": len(self.dataset.days),
            },
        )

    def demand_response(self):
        cfg = self.config.demand_response
        if not cfg.enabled:
            print("Demand response disabled", flush=True)
            return
        rows, baselines, shifts = [], [], []
        for day_idx, day in enumerate(self.dataset.days, start=1):
            houses = {i: self.dataset.frames[i][day] for i in self.config.players}
            aggregate = self.dataset.frames["prices"][day].copy()
            aggregate["consumption_kWh"] = sum(
                (df.consumption_kWh.to_numpy(float) for df in houses.values()),
                start=np.zeros(96),
            )
            aggregate["production_kWh"] = sum(
                (df.production_kWh.to_numpy(float) for df in houses.values()),
                start=np.zeros(96),
            )
            aggregate["net_load_kWh"] = (
                aggregate.consumption_kWh - aggregate.production_kWh
            )
            activation = dr.select_activation_window(
                aggregate, cfg.activation_periods, cfg.activation_metric
            )
            solved = {
                c: dr.solve_ideal_battery_day(aggregate, c)
                for c in sorted(
                    {0.0, *cfg.capacities_kwh, *cfg.reference_capacities_kwh}
                )
            }
            baselines += [
                {"day": day, "battery_capacity_kwh": c, "cost_eur": result["cost_eur"]}
                for c, result in solved.items()
            ]
            for capacity in cfg.capacities_kwh:
                baseline = solved[capacity]
                for mode in cfg.recovery_modes:
                    recovery = dr.select_recovery_window(
                        aggregate,
                        baseline["schedule"],
                        capacity,
                        activation,
                        cfg.recovery_periods,
                        mode,
                        cfg.allow_recovery_before_activation,
                    )
                    parameters = (
                        cfg.cap_ratios
                        if cfg.strategy == "cap"
                        else cfg.reduction_fractions
                    )
                    for parameter in parameters:
                        fn = (
                            dr.apply_cap_load_shift
                            if cfg.strategy == "cap"
                            else dr.apply_fractional_load_shift
                        )
                        modified, profile, players = fn(
                            aggregate, houses, activation, recovery, parameter
                        )
                        if (
                            abs(
                                modified.consumption_kWh.sum()
                                - aggregate.consumption_kWh.sum()
                            )
                            > 1e-7
                        ):
                            raise RuntimeError(
                                "Demand response must conserve total daily consumption"
                            )
                        result = dr.solve_ideal_battery_day(modified, capacity)
                        rows.append(
                            {
                                "day": day,
                                "battery_capacity_kwh": capacity,
                                "strategy": cfg.strategy,
                                "parameter": parameter,
                                "recovery_mode": mode,
                                "baseline_cost_eur": baseline["cost_eur"],
                                "dr_cost_eur": result["cost_eur"],
                                "saving_eur": baseline["cost_eur"] - result["cost_eur"],
                                "shifted_kwh": players.shifted_kwh.sum(),
                                "activation_start_period": activation[
                                    "activation_start_period"
                                ],
                                "recovery_start_period": recovery[
                                    "recovery_start_period"
                                ],
                            }
                        )
                        players = players.assign(
                            day=day,
                            battery_capacity_kwh=capacity,
                            recovery_mode=mode,
                            parameter=parameter,
                        )
                        shifts.extend(players.to_dict("records"))
            pd.DataFrame(rows).to_csv(
                self.output / "daily_demand_response.csv", index=False
            )
            print(f"Demand response: {day_idx}/{len(self.dataset.days)}", flush=True)
        pd.DataFrame(baselines).to_csv(
            self.output / "ideal_battery_reference.csv", index=False
        )
        pd.DataFrame(shifts).to_csv(self.output / "player_load_shifts.csv", index=False)
        summary = (
            pd.DataFrame(rows)
            .groupby(
                ["battery_capacity_kwh", "strategy", "parameter", "recovery_mode"],
                as_index=False,
            )[["baseline_cost_eur", "dr_cost_eur", "saving_eur", "shifted_kwh"]]
            .sum()
        )
        summary.to_csv(self.output / "demand_response_summary.csv", index=False)
        costs = pd.DataFrame(baselines).groupby("battery_capacity_kwh").cost_eur.sum()
        no_battery = float(costs.loc[0.0])
        comparisons = []
        for row in summary.to_dict("records"):
            for capacity in cfg.reference_capacities_kwh:
                full_cost = float(costs.loc[capacity])
                full_gain = no_battery - full_cost
                gap = float(row["baseline_cost_eur"]) - full_cost
                combined = no_battery - float(row["dr_cost_eur"])
                comparisons.append(
                    {
                        **row,
                        "reference_capacity_kwh": capacity,
                        "no_battery_cost_eur": no_battery,
                        "reference_cost_eur": full_cost,
                        "battery_only_saving_eur": no_battery
                        - row["baseline_cost_eur"],
                        "combined_saving_eur": combined,
                        "share_of_reference_gain_pct": 100 * combined / full_gain
                        if abs(full_gain) > 1e-10
                        else None,
                        "remaining_gap_closed_by_dr_pct": 100 * row["saving_eur"] / gap
                        if gap > 1e-10
                        else None,
                    }
                )
        pd.DataFrame(
            comparisons,
            columns=[
                *summary.columns,
                "reference_capacity_kwh",
                "no_battery_cost_eur",
                "reference_cost_eur",
                "battery_only_saving_eur",
                "combined_saving_eur",
                "share_of_reference_gain_pct",
                "remaining_gap_closed_by_dr_pct",
            ],
        ).to_csv(self.output / "comparison_to_full_battery.csv", index=False)

    def report(self):
        from .reporting import write_report

        write_report(self.output, self.config)


DEPENDENCIES = {
    "prepare": [],
    "representative": ["prepare"],
    "scores": ["representative"],
    "formation": ["scores"],
    "evaluate": ["formation"],
    "allocation": ["evaluate"],
    "shapley": ["prepare"],
    "stability": ["allocation"],
    "demand_response": ["prepare"],
    "report": ["evaluate"],
}


def run(config, base_dir, stage="all", output=None):
    experiment = Experiment(config, base_dir, output)
    wanted = ["allocation", "shapley", "demand_response", "report"]
    if config.stability:
        wanted.insert(1, "stability")
    if stage != "all":
        wanted = [stage]
    ordered = []

    def add(name):
        dependencies = (
            ["prepare"]
            if name == "formation" and config.partition_file
            else DEPENDENCIES[name]
        )
        for dep in dependencies:
            add(dep)
        if name not in ordered:
            ordered.append(name)

    if stage == "all":
        add("scores")
    for name in wanted:
        add(name)
    # Reports include all optional stages that ran before them.
    for name in ordered:
        print(f"\n[{name}]", flush=True)
        previous = json.loads(experiment.manifest.read_text())["completed_stages"]
        required = {
            "prepare": ["valid_days.csv", "data_audit.csv"],
            "representative": [
                "representative_days.csv",
                "aggregate_net_load_profiles.npy",
            ],
            "scores": ["pairwise_scores.csv", "pairwise_scores_long.csv"],
            "formation": ["selected_subcoalitions.csv", "formation_summary.json"],
            "evaluate": [
                "daily_evaluation.csv",
                "evaluation_summary.json",
                "daily_solo_costs.csv",
                "daily_subcoalition_costs.csv",
                "random_evaluation.csv",
            ],
            "allocation": ["annual_allocations.csv", "daily_allocations.csv"],
            "stability": ["stability_summary.json", "migration_candidates.csv"],
            "shapley": ["shapley_comparison.csv"] if config.shapley_players else [],
            "demand_response": [
                "demand_response_summary.csv",
                "comparison_to_full_battery.csv",
            ]
            if config.demand_response.enabled
            else [],
            "report": ["report.md", "plots/cumulative_cost.png"],
        }
        if (
            name in previous
            and name != stage
            and all((experiment.output / p).exists() for p in required[name])
        ):
            print(
                "Reusing completed stage with identical inputs and settings", flush=True
            )
            continue
        experiment.invalidate_stage(name)
        getattr(experiment, name)()
        experiment.record_stage(name)
    return experiment.output
