"""Prepared 15-minute CSV ingestion, conservative repair, and reproducible demo data."""

from pathlib import Path
from itertools import groupby
import hashlib
import numpy as np
import pandas as pd

T = 96
HOUSE_COLUMNS = ["consumption_kWh", "production_kWh"]
PRICE_COLUMNS = ["buy_price_eur_kwh", "sell_price_eur_kwh"]


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Dataset:
    def __init__(self, data_dir, players, repair=False, start_day=None, end_day=None):
        self.players = tuple(sorted(players))
        self.frames, self.audit, self.hashes = {}, [], {}
        names = {i: f"H{i}_15min.csv" for i in self.players}
        names["prices"] = "ireland_prices_15min.csv"
        for key, name in names.items():
            path = Path(data_dir) / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Supply prepared CSVs or use configs/demo.json."
                )
            self.hashes[name] = digest_file(path)
            raw = pd.read_csv(path)
            critical = PRICE_COLUMNS if key == "prices" else HOUSE_COLUMNS
            missing = set(["day", "period", *critical]) - set(raw.columns)
            if missing:
                raise ValueError(f"{name}: missing columns {sorted(missing)}")
            raw["day"] = pd.to_datetime(raw.day, errors="raise").dt.strftime("%Y-%m-%d")
            numeric_periods = pd.to_numeric(raw.period, errors="raise")
            if (
                not np.isfinite(numeric_periods).all()
                or not (numeric_periods == np.floor(numeric_periods)).all()
            ):
                raise ValueError(f"{name}: periods must be finite integers")
            raw["period"] = numeric_periods.astype(int)
            for col in critical:
                raw[col] = pd.to_numeric(raw[col], errors="raise")
            if start_day:
                raw = raw[raw.day >= str(pd.Timestamp(start_day).date())]
            if end_day:
                raw = raw[raw.day <= str(pd.Timestamp(end_day).date())]
            valid = {}
            for day, group in raw.groupby("day", sort=True):
                reason, repaired = "", False
                if group.period.duplicated().any():
                    reason = "duplicate_periods"
                elif not group.period.between(1, T).all():
                    reason = "period_out_of_range"
                else:
                    group = group.set_index("period").reindex(range(1, T + 1))
                    mask = group[critical].isna().any(axis=1).to_numpy()
                    total = int(mask.sum())
                    longest = max(
                        (len(list(g)) for v, g in groupby(mask) if v), default=0
                    )
                    if total and repair and total <= 4 and longest <= 2:
                        group[critical] = group[critical].interpolate(
                            method="linear", limit_direction="both"
                        )
                        repaired = True
                    if not np.isfinite(group[critical].to_numpy(float)).all():
                        reason = "unresolved_missing_or_nonfinite_values"
                    elif key != "prices" and (group[critical] < -1e-9).any().any():
                        reason = "negative_household_energy"
                    elif (
                        key == "prices"
                        and (
                            group.buy_price_eur_kwh < group.sell_price_eur_kwh - 1e-10
                        ).any()
                    ):
                        reason = "sell_price_exceeds_buy_price"
                if not reason:
                    group = group.reset_index()
                    group["day"] = day
                    group["date"] = pd.Timestamp(day) + pd.to_timedelta(
                        (group.period - 1) * 15, unit="m"
                    )
                    if key != "prices":
                        group["net_load_kWh"] = (
                            group.consumption_kWh - group.production_kWh
                        )
                    valid[day] = group
                self.audit.append(
                    {
                        "file": name,
                        "day": day,
                        "valid": not bool(reason),
                        "repaired": repaired,
                        "reason": reason,
                    }
                )
            self.frames[key] = valid
        all_days = set.union(*(set(v) for v in self.frames.values()))
        self.days = sorted(set.intersection(*(set(v) for v in self.frames.values())))
        if not self.days:
            raise ValueError(
                "No complete common days across the selected households and prices"
            )
        for row in self.audit:
            row["included_common_day"] = row["day"] in self.days
        self.excluded_days = sorted(all_days - set(self.days))

    def day_model(self, day, battery):
        from .model import DayModel

        return DayModel(
            day,
            {i: self.frames[i][day] for i in self.players},
            self.frames["prices"][day],
            battery,
        )

    def save_prepared(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for key, days in self.frames.items():
            name = (
                "ireland_prices_15min.csv" if key == "prices" else f"H{key}_15min.csv"
            )
            pd.concat([days[day] for day in self.days], ignore_index=True).to_csv(
                directory / name, index=False
            )

    def prosumers(self):
        return {
            i
            for i in self.players
            if sum(self.frames[i][d].production_kWh.sum() for d in self.days) > 1e-9
        }


def generate_demo(directory, players, days=6, seed=42):
    """Artificial profiles for exercising the pipeline, independent of thesis data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hours = np.arange(T) / 4
    rows = {i: [] for i in players}
    price_rows = []
    for d, date in enumerate(pd.date_range("2020-06-01", periods=days)):
        metadata = pd.DataFrame(
            {
                "date": date + pd.to_timedelta(np.arange(T) * 15, unit="m"),
                "day": str(date.date()),
                "period": np.arange(1, T + 1),
            }
        )
        buy = (
            0.18
            + 0.16 * np.exp(-(((hours - 19) / 2) ** 2))
            + 0.04 * np.exp(-(((hours - 8) / 2) ** 2))
        )
        prices = metadata.copy()
        prices["buy_price_eur_kwh"] = buy * (1 + 0.04 * np.sin(d))
        prices["sell_price_eur_kwh"] = 0.07
        price_rows.append(prices)
        for k, i in enumerate(players):
            frame = metadata.copy()
            load_kw = (
                0.3
                + 0.6 * np.exp(-(((hours - (7 + k % 3)) / 2) ** 2))
                + 1.5 * np.exp(-(((hours - (18 + k % 4)) / 2) ** 2))
            )
            load_kw *= 0.8 + 0.15 * (k % 4) + 0.08 * np.cos(d + k)
            load_kw += rng.uniform(0, 0.06, T)
            pv_kw = (
                np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
                * (2.5 + 0.5 * (k % 3))
                * (0.6 + 0.1 * (d % 4))
                if k < len(players) // 2
                else np.zeros(T)
            )
            frame["consumption_kWh"], frame["production_kWh"] = (
                load_kw * 0.25,
                pv_kw * 0.25,
            )
            frame["net_load_kWh"] = frame.consumption_kWh - frame.production_kWh
            rows[i].append(frame)
    for i, frames in rows.items():
        pd.concat(frames, ignore_index=True).to_csv(
            directory / f"H{i}_15min.csv", index=False
        )
    pd.concat(price_rows, ignore_index=True).to_csv(
        directory / "ireland_prices_15min.csv", index=False
    )
