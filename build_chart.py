#!/usr/bin/env python3
"""Build the S&P 500 return dispersion chart from local CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import yfinance as yf


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
DATA_DIR = WORKSPACE / "lse-dispersion" / "data"
OUTPUT = BASE / "dispersion chart 14.png"
NAMED_OUTPUT = BASE / "dispersion - weekly 1.0.png"
DESKTOP_OUTPUT = Path("/Users/ryan-desktop/Desktop/sp500-return-dispersion-chart-rolling-12m-weekly.png")
WEEKLY_CACHE = BASE / "weekly_adjusted_close_2013_2026.csv"
TOP_BOUND = 0.8


def jitter(index: pd.Series) -> pd.Series:
    return ((index * 37) % 100) / 100 * 0.72 - 0.36


def violin_jitter(values: pd.Series) -> pd.Series:
    scaled = (values.abs() / 0.8).clip(0, 1)
    width = 0.08 + (1 - scaled) * 0.34
    sequence = pd.Series(range(len(values)), index=values.index)
    offsets = ((sequence * 37) % 100) / 100 * 2 - 1
    return offsets * width


def boundary_y(values: pd.Series) -> pd.Series:
    sequence = pd.Series(range(len(values)), index=values.index)
    offsets = ((sequence * 37) % 100) / 100
    top = TOP_BOUND - 0.006 - offsets * 0.032
    bottom = -TOP_BOUND + 0.006 + offsets * 0.032
    return pd.Series(values.where(values >= 0, bottom).where(values < 0, top), index=values.index)


def load_weekly_prices(tickers: list[str]) -> pd.DataFrame:
    if WEEKLY_CACHE.exists():
        weekly = pd.read_csv(WEEKLY_CACHE, index_col=0, parse_dates=True)
    else:
        prices = yf.download(
            tickers,
            start="2013-01-01",
            end="2026-05-08",
            interval="1wk",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        weekly = prices["Adj Close"].dropna(axis=1, how="all")
        weekly.to_csv(WEEKLY_CACHE)

    return weekly.sort_index().resample("W-FRI").last()


def load_weekly_rolling_spread(weekly: pd.DataFrame) -> pd.DataFrame:
    rolling_returns = weekly / weekly.shift(52) - 1
    rolling_returns = rolling_returns.loc["2014-01-01":].dropna(how="all")
    stats = pd.DataFrame(
        {
            "date": rolling_returns.index,
            "count": rolling_returns.count(axis=1).values,
            "p10": rolling_returns.quantile(0.10, axis=1).values,
            "p90": rolling_returns.quantile(0.90, axis=1).values,
        }
    )
    stats = stats[stats["count"] >= 100].copy()
    stats["spread_p90_p10"] = stats["p90"] - stats["p10"]
    return stats


def main() -> None:
    annual = pd.read_csv(DATA_DIR / "sp500_spy_503_annual_returns_2010_2026.csv")
    tickers = sorted(annual["ticker"].unique())
    weekly = load_weekly_prices(tickers)

    rolling_52w = weekly / weekly.shift(52) - 1
    latest_2026 = rolling_52w.dropna(how="all").iloc[-1].dropna()
    latest_2026 = latest_2026.rename("return").reset_index().rename(columns={"index": "ticker"})
    latest_2026["year"] = 2026

    annual = annual[(annual["year"] >= 2014) & (annual["year"] <= 2026)].copy()
    annual = annual[annual["year"] != 2026]
    annual = pd.concat([annual[["ticker", "year", "return"]], latest_2026[["ticker", "year", "return"]]], ignore_index=True)
    annual["market_avg"] = annual.groupby("year")["return"].transform("mean")
    annual["diff_from_market"] = annual["return"] - annual["market_avg"]
    annual["plot_diff"] = annual["diff_from_market"].clip(-TOP_BOUND, TOP_BOUND)
    annual["out_of_range"] = annual["diff_from_market"] != annual["plot_diff"]
    annual["plot_y"] = annual["plot_diff"]
    out_of_range = annual["out_of_range"]
    annual.loc[out_of_range, "plot_y"] = annual.groupby(["year", annual["diff_from_market"].gt(0)])["plot_diff"].transform(boundary_y)
    annual = annual.sort_values(["year", "ticker"]).reset_index(drop=True)

    spread = load_weekly_rolling_spread(weekly)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 9.6), dpi=300)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.35, 1.25], hspace=0.20)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    ax_top.set_facecolor("#f8fafc")
    x = annual["year"] + annual.groupby("year")["plot_diff"].transform(violin_jitter)
    in_range = ~annual["out_of_range"]
    ax_top.scatter(
        x[in_range],
        annual.loc[in_range, "plot_y"],
        s=9,
        alpha=0.48,
        color="#4f83e6",
        edgecolors="none",
        rasterized=True,
    )
    ax_top.scatter(
        x[~in_range],
        annual.loc[~in_range, "plot_y"],
        s=10,
        alpha=0.76,
        color="#2f63c7",
        edgecolors="none",
        rasterized=True,
    )
    ax_top.axhline(0, color="#111827", linewidth=1.6)
    ax_top.set_xlim(2013.35, 2026.65)
    ax_top.set_ylim(-0.8, 0.8)
    ax_top.set_title("S&P 500 Return Dispersion", loc="left", fontsize=24, weight="bold", pad=16)
    ax_top.set_ylabel("% diff from market avg.", fontsize=11, labelpad=8)
    ax_top.set_xticks(range(2014, 2027))
    ax_top.set_xticklabels([str(year) for year in range(2014, 2027)], fontsize=12, fontweight="bold")
    ax_top.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_top.tick_params(axis="y", labelsize=10)
    ax_top.grid(True, axis="y", color="#d8e0ea", linewidth=0.8)
    ax_top.grid(True, axis="x", color="#e8edf4", linewidth=0.55)

    ax_bottom.set_facecolor("#fbfbfc")
    ax_bottom.fill_between(
        spread["date"],
        spread["spread_p90_p10"],
        0.05,
        color="#f4d8ad",
        alpha=0.68,
        linewidth=0,
    )
    ax_bottom.plot(
        spread["date"],
        spread["spread_p90_p10"],
        color="#8f3d17",
        linewidth=2.1,
    )
    ax_bottom.set_title("Spread: Top vs Bottom 10%", loc="left", fontsize=13, weight="bold", pad=8)
    ax_bottom.set_ylim(0.40, 1.30)
    ax_bottom.set_ylabel("Spread Top vs. Bottom 10%", fontsize=10, labelpad=18)
    ax_bottom.set_xlim(pd.Timestamp("2013-05-08"), pd.Timestamp("2026-08-27"))
    ax_bottom.set_yticks([value / 100 for value in range(40, 131, 10)])
    year_ticks = pd.date_range("2014-01-01", "2026-01-01", freq="YS")
    ax_bottom.set_xticks(year_ticks)
    ax_bottom.set_xticklabels([str(date.year) for date in year_ticks], fontsize=12, fontweight="bold")
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_bottom.tick_params(axis="y", labelsize=10)
    ax_bottom.grid(True, color="#dde4ec", linewidth=0.8)

    for axis in (ax_top, ax_bottom):
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#111827")

    fig.subplots_adjust(left=0.135, right=0.986, top=0.91, bottom=0.085)
    fig.savefig(OUTPUT, dpi=300, facecolor="white")
    fig.savefig(NAMED_OUTPUT, dpi=300, facecolor="white")
    fig.savefig(DESKTOP_OUTPUT, dpi=300, facecolor="white")
    print(OUTPUT)
    print(NAMED_OUTPUT)
    print(DESKTOP_OUTPUT)


if __name__ == "__main__":
    main()
