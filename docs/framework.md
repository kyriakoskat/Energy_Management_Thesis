# Framework guide

## Inputs

The runner consumes prepared 15-minute CSVs from `data_dir`:

| File | Required columns |
| --- | --- |
| `H<id>_15min.csv` | `day`, `period`, `consumption_kWh`, `production_kWh` |
| `ireland_prices_15min.csv` | `day`, `period`, `buy_price_eur_kwh`, `sell_price_eur_kwh` |

`day` identifies the calendar date. Periods are integers 1–96. Consumption and production represent energy during each interval. The loader derives timestamps from day/period and recomputes `net_load_kWh = consumption_kWh - production_kWh`.

The common evaluation horizon contains complete days present for every configured participant and the prices. Duplicate periods, nonfinite values and negative household energy exclude a day. The cooperative scheduling model uses buy prices greater than or equal to sell prices. Each exclusion appears in `data_audit.csv`.

Set `repair_small_gaps` to `true` to apply the thesis's conservative interpolation rule: at most two consecutive missing periods and at most four missing periods in a day. The loader interpolates critical values within the day, fills permitted boundary gaps with the nearest value and retains only complete common days. The default accepts the existing cleaned dataset directly.

## Stages and outputs

| Stage | Main outputs |
| --- | --- |
| `prepare` | `prepared_data/`, `valid_days.csv`, `data_audit.csv` |
| `representative` | `representative_days.csv`, cluster assignments, initialization logs |
| `scores` | Pair/day costs, weighted pairwise score matrix |
| `formation` | `selected_subcoalitions.csv`, solver status or local-search history |
| `evaluate` | Daily costs, annual totals, balanced random-partition comparisons |
| `allocation` | Daily and annual player bonuses and cost shares within selected groups |
| `shapley` | Daily comparison on `shapley_players` over the configured benchmark horizon |
| `stability` | Weakest players, migration candidates and destination-member impacts |
| `demand_response` | Load shifts, savings by capacity and comparison with large-battery references |
| `report` | `report.md` and PNG figures |

`--stage all` runs the complete configured workflow. `--stage <name>` runs that stage and its prerequisites. Optional Shapley and DR settings control which experiments run; `stability` controls inclusion of migration testing in the full workflow. An explicitly selected `stability` stage still performs that test.

For a supplied `partition_file`, formation validates and loads that grouping. A direct evaluation can then begin from the prepared data without recomputing pairwise scores. The complete workflow still computes the representative-day scores for inspection.

## Scheduling and economics

The common optimizer preserves the distributed household model used in the original experiments:

- 96 intervals with a duration of 0.25 hours.
- Per-household batteries with 10 kWh nominal capacity, 3.3 kW power, 95% roundtrip efficiency and 85% usable capacity.
- Initial SOC at 50% of usable capacity and matching final SOC.
- Grid import/export limits of 20 kW per household.
- Battery throughput degradation, weighted by 0.5, using BLCOE 0.05 EUR/kWh.
- Directed internal energy trades with the original `trade_epsilon = 0.001` regularization applied to the sum of trade-power variables.

Individual scheduling excludes simultaneous grid import/export. Cooperative scheduling retains the original formulation. Scheduling results must reach the solver's `OPTIMAL` status. These assumptions are configurable in the `battery` section.

For coalition S, the cooperation-surplus game is

```text
g(S) = sum of members' optimized solo costs - optimized coalition cost.
m_i  = g(S) - g(S without i).
b_i  = g(S) * max(m_i, 0) / sum_j max(m_j, 0).
allocated_cost_i = solo_cost_i - b_i.
```

The fallback splits a positive surplus equally when every contribution weight is zero. Exact Shapley uses the same cooperation-surplus game on a subset of at most ten participants. The thesis configuration uses players 1, 3, 5, 17, 14, 18, 19 and 20 over June 2020.

## Representative days and group formation

Two distinct extremes capture maximum aggregate positive net load and maximum aggregate surplus. K-medoids clusters the remaining standardized 96-period profiles using 50 initializations and a fixed seed. A medoid's weight is its cluster size; each extreme has weight one. The default twelve medoids plus two extremes represent the entire common-day horizon. Short demos reduce the number of clusters to the available non-extreme days.

Pairwise scores sum weighted daily gains. Group formation first maximizes the weakest group's internal pair score, then total internal score subject to that minimum. The MILP reports both stage statuses and whether both optima were certified. `local_search` performs multiple random starts with improving player swaps and reports a heuristic result. `auto` selects MILP up to `exact_max_players`, then local search. Balanced random comparison partitions distribute prosumers as evenly as group size permits.

Migration testing aggregates player bonuses over the evaluation horizon, picks each group's weakest member and tests joining every other existing group. A move is accepted if the mover improves and no existing destination member loses, using a 1e-8 EUR tolerance. The criterion concerns these specific individual moves; source-group members do not exercise a veto.

## Demand response benchmark

This stage retains the thesis's separate central ideal-battery model: perfect efficiency, empty initial/final SOC and no power or degradation limits. It aggregates household loads and PV before scheduling. Its no-battery reference therefore includes internal matching of local production and demand.

The default policy caps each household's activation-window energy at 80% of the mean household energy in that window. Excess consumption shifts proportionally from the activation window into a nonoverlapping recovery window, with total daily consumption conserved. Recovery can use `battery_aware`, `cheapest_price` or `surplus_first` ranking. The runner also supports the original fractional-shift policy.

`comparison_to_full_battery.csv` reports battery-only savings, incremental DR savings, combined savings, the share of the large-battery reference gain and the fraction of the remaining gap closed by DR.

## Configuration and resumption

Configuration paths are relative to the JSON file. `--output` is relative to the current working directory and overrides `output_dir`.

`manifest.json` records settings, Python/package versions, common days and input SHA-256 hashes. The run fingerprint also includes framework/cache versions and the hash of a supplied partition. Completed stages and scalar coalition costs are reused for matching settings and inputs. Rerunning a stage invalidates completion records for dependent results; interrupted work retains completed coalition solves. Use a separate output directory when changing an experiment.

For the package versions used during verification:

```bash
python -m pip install -r requirements-verified.txt
python -m pip install -e . --no-deps
```

## Original source mapping

| Original experiment family | Current module |
| --- | --- |
| `01_dataset_audit`, `01c_repair_small_missing_gaps_v3`, `01d_finalize_clean_dataset_drop_unresolved_days` | `data.py` |
| `run_coalition_real_data`, `subcoalition_decomposition_experiment`, scheduling portions of `03` and `05` | `model.py` |
| `02_select_extreme_and_kmedoids_days` | `clustering.py`, representative-day stage |
| `03_build_pairwise_scores_from_representative_days` | Pairwise score stage |
| `04_form_subcoalitions_lexicographic_only_30_people_full`, `04_form_subcoalitions_pairwise_local_search_*` | `formation.py` |
| `05_annual_evaluation_*` | Evaluation stage |
| `06_full_year_accepted_migration_stability_*` | Stability stage |
| `09_demand_response_targeted_load_shift` | `demand_response.py`, DR stage |

The final thesis and presentation provide the scientific context. Source hashes for the refactoring are recorded in `source_provenance.json`.
