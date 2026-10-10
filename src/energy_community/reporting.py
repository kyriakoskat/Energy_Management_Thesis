"""CSV-backed result summary and figures for a completed experiment."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_report(output, config):
    output = Path(output)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    daily = pd.read_csv(output / "daily_evaluation.csv")
    completed = json.loads((output / "manifest.json").read_text()).get(
        "completed_stages", []
    )
    totals = json.loads((output / "evaluation_summary.json").read_text())
    dates = pd.to_datetime(daily.day)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, label, color in [
        ("baseline_cost_eur", "Passive baseline", "#9ba5b1"),
        ("solo_cost_eur", "Individual batteries", "#cc9b35"),
        ("grand_cost_eur", "Grand coalition", "#185b91"),
        ("subcoalitions_cost_eur", "Subcoalitions", "#419487"),
    ]:
        ax.plot(dates, daily[column].cumsum(), label=label, color=color, lw=2)
    ax.set(
        ylabel="Cumulative cost (EUR)",
        title="Energy management over the evaluation horizon",
    )
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots / "cumulative_cost.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(
        dates,
        daily.grand_coordination_gain_eur.cumsum(),
        label="Grand coalition",
        color="#185b91",
        lw=2,
    )
    ax.plot(
        dates,
        daily.subcoalitions_coordination_gain_eur.cumsum(),
        label="Selected subcoalitions",
        color="#419487",
        lw=2,
    )
    random_path = output / "random_evaluation.csv"
    if random_path.exists():
        random = pd.read_csv(random_path)
        for k, frame in random.groupby("partition_id"):
            ax.plot(
                pd.to_datetime(frame.day),
                frame.coordination_gain_eur.cumsum(),
                color="#b4bac1",
                alpha=0.6,
                label="Balanced random partitions" if k == 1 else None,
            )
    ax.set(
        ylabel="Cumulative cooperation gain (EUR)",
        title="Gain beyond individual battery scheduling",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots / "coordination_gain.png")
    plt.close(fig)
    if "scores" in completed and (output / "pairwise_scores.csv").exists():
        scores = pd.read_csv(output / "pairwise_scores.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(scores.to_numpy(), cmap="Blues")
        if len(scores) <= 30:
            ax.set_xticks(np.arange(len(scores)), scores.columns, rotation=90)
            ax.set_yticks(np.arange(len(scores)), scores.index)
        ax.set(
            xlabel="Player",
            ylabel="Player",
            title="Weighted pairwise cooperation scores",
        )
        fig.colorbar(im, ax=ax, label="Weighted gain (EUR)")
        fig.tight_layout()
        fig.savefig(plots / "pairwise_scores.png")
        plt.close(fig)
    if "representative" in completed and (output / "representative_days.csv").exists():
        reps = pd.read_csv(output / "representative_days.csv")
        days = pd.read_csv(output / "valid_days.csv").day.tolist()
        profiles = np.load(output / "aggregate_net_load_profiles.npy")
        fig, ax = plt.subplots(figsize=(9, 4))
        for row in reps.itertuples():
            ax.plot(
                np.arange(96) / 4,
                profiles[days.index(row.day)],
                lw=1.4,
                label=f"{row.day} ({row.kind}, w={row.weight})",
            )
        ax.axhline(0, color="#777", lw=0.6)
        ax.set(
            xlabel="Hour",
            ylabel="Aggregate net load (kWh / 15 min)",
            title="Representative days",
        )
        ax.legend(
            frameon=False,
            fontsize=7,
            ncol=2,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        fig.tight_layout()
        fig.savefig(plots / "representative_days.png")
        plt.close(fig)
    allocation_path = output / "annual_allocations.csv"
    if "allocation" in completed and allocation_path.exists():
        table = pd.read_csv(allocation_path).sort_values("player_id")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(table.player_id.astype(str), table.allocated_bonus, color="#185b91")
        ax.set(
            xlabel="Player",
            ylabel="Allocated bonus (EUR)",
            title="Positive marginal contribution allocation",
        )
        fig.tight_layout()
        fig.savefig(plots / "allocated_bonus.png")
        plt.close(fig)
    dr_path = output / "demand_response_summary.csv"
    if (
        "demand_response" in completed
        and config.demand_response.enabled
        and dr_path.exists()
    ):
        table = pd.read_csv(dr_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        for (strategy, parameter, mode), group in table.groupby(
            ["strategy", "parameter", "recovery_mode"]
        ):
            ax.plot(
                group.battery_capacity_kwh,
                group.saving_eur,
                marker="o",
                label=f"{strategy}={parameter:g}, {mode}",
            )
        ax.set(
            xlabel="Ideal central battery capacity (kWh)",
            ylabel="Additional DR saving (EUR)",
            title="Targeted demand response",
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plots / "demand_response.png")
        plt.close(fig)
    lines = [
        f"# {config.name}",
        "",
        f"Data: {'synthetic demonstration' if config.demo else 'user-supplied prepared dataset'}.",
        f"Participants: {len(config.players)}. Common evaluation days: {len(daily)}.",
        "",
        "| Operating mode | Total cost (EUR) |",
        "| --- | ---: |",
    ]
    for column, label in [
        ("baseline_cost_eur", "Passive baseline"),
        ("solo_cost_eur", "Individual scheduling"),
        ("grand_cost_eur", "Grand coalition"),
        ("subcoalitions_cost_eur", "Selected subcoalitions"),
    ]:
        lines.append(f"| {label} | {totals[column]:.6f} |")
    recovery = totals["coordination_gain_recovery_pct"]
    if recovery is not None:
        lines += [
            "",
            f"Selected groups recover {recovery:.2f}% of the grand coalition coordination gain.",
        ]
    lines += [
        "",
        "![Cumulative costs](plots/cumulative_cost.png)",
        "",
        "![Cooperation gains](plots/coordination_gain.png)",
    ]
    if "scores" in completed and (plots / "pairwise_scores.png").exists():
        lines += ["", "![Pairwise scores](plots/pairwise_scores.png)"]
    stability_path = output / "stability_summary.json"
    if "stability" in completed and stability_path.exists():
        summary = json.loads(stability_path.read_text())
        lines += [
            "",
            f"Accepted profitable migrations: {summary['accepted_profitable_moves']} / {summary['candidate_moves']}, under the weakest-player migration criterion.",
        ]
    if (
        "demand_response" in completed
        and config.demand_response.enabled
        and dr_path.exists()
    ):
        lines += [
            "",
            "![Demand response](plots/demand_response.png)",
            "",
            "The DR experiment uses the ideal central-battery benchmark defined in the thesis, with perfect efficiency and no power or degradation limits.",
        ]
    lines += [
        "",
        "Configuration, software versions and input SHA-256 hashes are recorded in `manifest.json`.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
