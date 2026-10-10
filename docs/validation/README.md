# Validation

Verification used Python 3.12 with OR-Tools 9.15.6755, NumPy 2.3.5, pandas 2.2.3 and Matplotlib 3.10.8.

- A six-day, six-player synthetic demonstration exercised the full workflow, including grouping, allocations, exact Shapley, weakest-player migrations and both battery-aware and price-based demand response.
- The test suite covers physical scheduling constraints, economic allocation identities, exact grouping versus exhaustive partitions, conservative data repair, DR energy conservation and interrupted-run recovery.
- Recorded costs for the original 20-player partition were recomputed on 2020-01-02, 2020-06-15 and 2020-12-11. Passive, solo, grand-coalition and subcoalition costs agree within 1e-5 EUR. The observed maximum absolute difference is below 3e-13 EUR.
- The passive baseline across all 343 available clean days reproduces €7,259.680763625023. Annual optimized totals in `docs/reference_results/` are the original recorded thesis outputs; the optimized regression evaluates the three specified dates.

The numerical comparisons are stored in `recorded_result_regression.csv` and `annual_baseline_check.json`.

The optional recorded-results test can be rerun with the prepared research data:

```bash
# Linux/macOS
ENERGY_THESIS_DATA_DIR=/path/to/processed_data_clean python -m pytest -q
```

In PowerShell, set `$env:ENERGY_THESIS_DATA_DIR` before running pytest. The public suite runs using generated synthetic data.
